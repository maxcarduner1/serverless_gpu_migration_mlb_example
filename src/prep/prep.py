"""Prep task (serverless CPU).

Roboflow Universe datasets  ->  /Volumes/<catalog>/<schema>/raw_images

1. Download each public dataset version ONCE, as a YOLOv8 export zip, into
   raw_images/sources/. If the zip is already there, skip the download. This
   keeps Roboflow free-tier usage minimal.
2. Merge the classes into one label space: 0=ball, 1=bat, 2=glove.
3. Subsample to a capped size (default about 1,800 train, 300 val, 60 test).
4. Write a YOLO-format dataset plus data.yaml to raw_images/yolo/.
5. Write a Delta manifest table <catalog>.<schema>.yolo_manifest with split,
   per-image class counts, source dataset, and license.

The Roboflow API key is read from the `roboflow` secret scope, and only
here. The GPU task reads the curated data from the Volume and needs no
secrets.
"""

import argparse
import os
import random
import shutil
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import requests
import yaml

TARGET_CLASSES = ["ball", "bat", "glove"]

# Source-class name (lower-case) -> target class name. Unmapped classes are dropped.
CLASS_MAP = {
    "baseball": "ball",
    "ball": "ball",
    "bat": "bat",
    "glove": "glove",
}

# Public Roboflow Universe datasets. The license is recorded per row in the manifest.
DATASETS = [
    {
        "workspace": "kschattin",
        "project": "pitch-tracking",
        "license": "MIT",
        "url": "https://universe.roboflow.com/kschattin/pitch-tracking",
    },
    {
        "workspace": "baseball-94erg",
        "project": "baseball-detection-v1-vvupe",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/baseball-94erg/baseball-detection-v1-vvupe",
    },
    {
        "workspace": "tel-4fuev",
        "project": "baseball-glove-j2hci",
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/tel-4fuev/baseball-glove-j2hci",
    },
]

API = "https://api.roboflow.com"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="raw_images")
    p.add_argument("--max_train", type=int, default=1800, help="total across datasets")
    p.add_argument("--max_val", type=int, default=300)
    p.add_argument("--max_test", type=int, default=60)
    p.add_argument("--min_version_images", type=int, default=400,
                   help="pick the smallest dataset version with at least this many images")
    p.add_argument("--seed", type=int, default=42)
    # parse_known_args: ignore anything extra the Jobs runtime passes through
    return p.parse_known_args()[0]


def get_api_key():
    from databricks.sdk.runtime import dbutils

    return dbutils.secrets.get(scope="roboflow", key="roboflow_api_key")


# --------------------------------------------------------------------------- Roboflow
class RoboflowError(RuntimeError):
    """Roboflow HTTP failure with a sanitized message (no API key, no signed URLs)."""


def _redact(url):
    """Drop the query string. It holds the api_key, or the signature of a signed download link."""
    return str(url).split("?", 1)[0]


def rf_get(url, api_key=None, *, what, **kwargs):
    """GET from Roboflow. Errors are re-raised without the API key or signed URLs.

    requests puts the full request URL (including `?api_key=...`) into the text of HTTPError and
    ConnectionError. So we catch every RequestException and raise a new error built only from
    the status code and the redacted URL. `from None` drops the original exception, which holds
    the key, from the traceback.
    """
    params = {"api_key": api_key} if api_key else None
    try:
        r = requests.get(url, params=params, **kwargs)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        raise RoboflowError(
            f"Roboflow request failed ({what}): {type(e).__name__}"
            f"{f' HTTP {status}' if status else ''} for {_redact(url)}"
        ) from None


def pick_version(ds, api_key, min_images):
    """Return the smallest version that still has >= min_images (fewer bytes to download)."""
    r = rf_get(f"{API}/{ds['workspace']}/{ds['project']}", api_key, what="list versions", timeout=60)
    versions = r.json().get("versions", [])
    if not versions:
        raise RuntimeError(f"No dataset versions published for {ds['url']}")
    cands = []
    for v in versions:
        num = int(str(v["id"]).rstrip("/").split("/")[-1])
        cands.append((int(v.get("images") or 0), num))
    big_enough = sorted(c for c in cands if c[0] >= min_images)
    images, num = big_enough[0] if big_enough else max(cands)
    print(f"[{ds['project']}] versions (images, v) = {sorted(cands)} -> using v{num} ({images} images)")
    return num


def download_once(ds, api_key, sources_dir, min_images):
    """Download a YOLOv8 export zip into the Volume unless one already exists."""
    slug = f"{ds['workspace']}__{ds['project']}"
    existing = sorted(sources_dir.glob(f"{slug}__v*.zip"))
    if existing:
        print(f"[{ds['project']}] already downloaded -> {existing[-1].name} (skipping Roboflow call)")
        return existing[-1]

    version = pick_version(ds, api_key, min_images)
    url = f"{API}/{ds['workspace']}/{ds['project']}/{version}/yolov8"
    for _ in range(60):  # Roboflow may need to generate the export first
        r = rf_get(url, api_key, what="request export", timeout=120)
        link = r.json().get("export", {}).get("link")  # signed URL: never printed
        if link:
            break
        print(f"[{ds['project']}] export being generated ... {r.json().get('progress')}")
        time.sleep(10)
    else:
        raise TimeoutError(f"Roboflow export did not become ready for {_redact(url)}")

    target = sources_dir / f"{slug}__v{version}.zip"
    tmp = Path(tempfile.mkdtemp()) / target.name
    resp = rf_get(link, what="download export", stream=True, timeout=600)
    try:
        with resp, open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    except requests.RequestException as e:  # mid-stream failure: hide the signed link
        raise RoboflowError(f"Roboflow export download interrupted: {type(e).__name__}") from None
    shutil.copyfile(tmp, target)  # write the final file only once it is complete
    print(f"[{ds['project']}] downloaded {tmp.stat().st_size / 1e6:.1f} MB -> {target}")
    return target


# --------------------------------------------------------------------------- labels
def to_box(vals):
    """YOLO bbox (cx cy w h) or polygon (x1 y1 x2 y2 ...) -> bbox."""
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def remap_labels(label_file, id_map):
    """Return remapped YOLO label lines for one image (only target classes)."""
    lines = []
    if not label_file.exists():
        return lines
    for raw in label_file.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        src = int(float(parts[0]))
        if src not in id_map:
            continue
        box = to_box([float(x) for x in parts[1:]])
        lines.append(f"{id_map[src]} " + " ".join(f"{v:.6f}" for v in box))
    return lines


def collect(extract_dir):
    """Map split -> list of (image_path, label_path) from a Roboflow YOLOv8 export."""
    meta = yaml.safe_load((extract_dir / "data.yaml").read_text())
    names = meta["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    id_map = {}
    for i, n in enumerate(names):
        tgt = CLASS_MAP.get(str(n).strip().lower())
        if tgt:
            id_map[i] = TARGET_CLASSES.index(tgt)
    print(f"  source classes {names} -> mapped {id_map}")

    splits = {}
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        img_dir = extract_dir / folder / "images"
        if img_dir.exists():
            imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
            splits[split] = [(p, extract_dir / folder / "labels" / f"{p.stem}.txt") for p in imgs]
    return splits, id_map


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    rng = random.Random(args.seed)
    vol = Path(f"/Volumes/{args.catalog}/{args.schema}/{args.volume}")
    if not vol.exists():
        raise FileNotFoundError(f"{vol} not found. Create the volume first (see README).")
    sources_dir = vol / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    out_dir = vol / "yolo"

    api_key = None
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    per_ds = {k: v // len(DATASETS) for k, v in caps.items()}

    # Build the curated dataset locally, then copy it to the Volume in one pass.
    stage = Path(tempfile.mkdtemp()) / "yolo"
    rows = []
    for ds in DATASETS:
        if not any(sources_dir.glob(f"{ds['workspace']}__{ds['project']}__v*.zip")):
            api_key = api_key or get_api_key()
        zpath = download_once(ds, api_key, sources_dir, args.min_version_images)
        extract = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(zpath) as z:
            z.extractall(extract)
        splits, id_map = collect(extract)

        # Keep source splits where they exist. Missing val/test are carved out of train.
        pool_train = splits.get("train", [])
        rng.shuffle(pool_train)
        for split in ["test", "val"]:
            if not splits.get(split):
                n = per_ds[split] * 2
                splits[split], pool_train = pool_train[:n], pool_train[n:]
        splits["train"] = pool_train

        for split, items in splits.items():
            rng.shuffle(items)
            kept = 0
            for img, lbl in items:
                if kept >= per_ds[split]:
                    break
                lines = remap_labels(lbl, id_map)
                if not lines:  # drop images with no target-class boxes
                    continue
                name = f"{ds['project']}__{img.stem}{img.suffix.lower()}"
                (stage / split / "images").mkdir(parents=True, exist_ok=True)
                (stage / split / "labels").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(img, stage / split / "images" / name)
                (stage / split / "labels" / f"{Path(name).stem}.txt").write_text("\n".join(lines) + "\n")
                counts = [sum(1 for l in lines if l.startswith(f"{c} ")) for c in range(len(TARGET_CLASSES))]
                rows.append({
                    "split": split,
                    "image_path": str(out_dir / split / "images" / name),
                    "source_dataset": f"{ds['workspace']}/{ds['project']}",
                    "source_version": zpath.stem.split("__v")[-1],
                    "license": ds["license"],
                    "attribution_url": ds["url"],
                    "n_ball": counts[0],
                    "n_bat": counts[1],
                    "n_glove": counts[2],
                })
                kept += 1
            print(f"  {split}: kept {kept}")
        shutil.rmtree(extract, ignore_errors=True)

    data_yaml = {"path": str(out_dir), "train": "train/images", "val": "val/images",
                 "test": "test/images", "names": dict(enumerate(TARGET_CLASSES))}
    (stage / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # Replace the curated dataset atomically-ish (raw zips in sources/ are never touched).
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(stage, out_dir)
    print(f"Wrote {len(rows)} images + data.yaml to {out_dir}")

    # Also write the curated set as ONE tar file. Reading thousands of small files through
    # the Volume mount is slow; the GPU task copies this single file and extracts it locally.
    tar_local = stage.parent / "yolo_dataset.tar"
    with tarfile.open(tar_local, "w") as tar:
        tar.add(stage, arcname="yolo")
    shutil.copyfile(tar_local, vol / "yolo_dataset.tar")
    print(f"Wrote {vol / 'yolo_dataset.tar'} ({tar_local.stat().st_size / 1e6:.1f} MB)")

    # Delta manifest
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    table = f"{args.catalog}.{args.schema}.yolo_manifest"
    spark.createDataFrame(rows).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    summary = spark.sql(f"""
        SELECT split, source_dataset, license, count(*) AS images,
               sum(n_ball) AS ball, sum(n_bat) AS bat, sum(n_glove) AS glove
        FROM {table} GROUP BY ALL ORDER BY split, source_dataset""")
    summary.show(truncate=False)
    print(f"Manifest table: {table}")


if __name__ == "__main__":
    main()
