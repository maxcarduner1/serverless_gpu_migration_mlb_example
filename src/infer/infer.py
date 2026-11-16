"""Batch inference task (Serverless GPU, AI env v6): YOLO detection on the held-out test split.

* Loads the model registered by the training task: models:/<catalog>.<schema>.<model>@champion
* Writes annotated images, a montage, and a short GIF to /Volumes/<catalog>/<schema>/artifacts/inference/<ts>/
* Writes detections to predictions.parquet in the same folder. The `publish` CPU task then
  appends them to the Delta table <catalog>.<schema>.yolo_predictions.
* Logs detection counts and the montage/GIF to the launcher's MLflow run.

Runs in the AI environment (torch/ultralytics preinstalled), the same as training.
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="cmegdemos_catalog")
    p.add_argument("--schema", default="mlb_air_migration")
    p.add_argument("--model_name", default="mlb_yolo_detector")
    p.add_argument("--alias", default="champion")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max_images", type=int, default=60)
    return p.parse_known_args()[0]


def montage(images, cols=4, tile=(400, 300)):
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile[0], rows * tile[1]), "black")
    for i, im in enumerate(images):
        sheet.paste(im.resize(tile), ((i % cols) * tile[0], (i // cols) * tile[1]))
    return sheet


def main():
    args = parse_args()
    vol = Path(f"/Volumes/{args.catalog}/{args.schema}")
    test_dir = vol / "raw_images" / "yolo" / "test" / "images"
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = vol / "artifacts" / "inference" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    full_name = f"{args.catalog}.{args.schema}.{args.model_name}"
    mlflow.set_registry_uri("databricks-uc")
    version = mlflow.MlflowClient().get_model_version_by_alias(full_name, args.alias).version
    uri = f"models:/{full_name}/{version}"
    print(f"Loading {uri} (@{args.alias})", flush=True)
    yolo = mlflow.pyfunc.load_model(uri).unwrap_python_model().model  # ultralytics YOLO

    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXT)[: args.max_images]
    rows, annotated = [], []
    t0 = time.time()
    for path in images:
        r = yolo.predict(str(path), conf=args.conf, verbose=False, device=0)[0]
        im = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB
        im.save(out_dir / f"{path.stem}_pred.jpg", quality=90)
        annotated.append(im)
        for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            rows.append({
                "image_path": str(path), "class_id": int(c), "class_name": r.names[int(c)],
                "confidence": float(s), "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "model": full_name, "model_version": str(version), "inference_ts": ts,
            })
    secs = time.time() - t0
    print(f"{len(images)} images -> {len(rows)} detections in {secs:.1f}s", flush=True)

    if annotated:
        montage(annotated[:12]).save(out_dir / "montage.jpg", quality=90)
        frames = [im.resize((640, int(640 * im.height / im.width))) for im in annotated[:20]]
        frames[0].save(out_dir / "demo.gif", save_all=True, append_images=frames[1:], duration=700, loop=0)

    df = pd.DataFrame(rows, columns=["image_path", "class_id", "class_name", "confidence", "x1", "y1",
                                     "x2", "y2", "model", "model_version", "inference_ts"])
    df.to_parquet(out_dir / "predictions.parquet", index=False)
    (vol / "artifacts" / "inference" / "LATEST").write_text(ts)
    print(f"Annotated images, montage.jpg, demo.gif, predictions.parquet -> {out_dir}", flush=True)

    counts = df["class_name"].value_counts().to_dict()
    with mlflow.start_run():  # resumes the launcher's run (MLFLOW_RUN_ID)
        mlflow.log_params({"model_uri": uri, "conf": args.conf, "n_images": len(images)})
        mlflow.log_metrics({"n_detections": len(rows), "seconds": secs,
                            **{f"detections_{k}": v for k, v in counts.items()}})
        for f in ["montage.jpg", "demo.gif"]:
            if (out_dir / f).exists():
                mlflow.log_artifact(str(out_dir / f))
    print(json.dumps({"inference_ts": ts, "detections": counts}), flush=True)


if __name__ == "__main__":
    main()
