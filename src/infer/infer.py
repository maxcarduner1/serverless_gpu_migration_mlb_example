"""Inference task (serverless CPU): batch YOLO detection on the held-out test split.

* Loads the model registered by the GPU task: models:/<catalog>.<schema>.<model>@champion
* Writes annotated images, a montage, and a short GIF to /Volumes/<catalog>/<schema>/artifacts/inference/<ts>/
* Appends detections to the Delta table <catalog>.<schema>.yolo_predictions
"""

import argparse
import time
from pathlib import Path

import mlflow
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
    print(f"Loading {uri} (@{args.alias})")
    yolo = mlflow.pyfunc.load_model(uri).unwrap_python_model().model  # ultralytics YOLO

    images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in IMG_EXT)[: args.max_images]
    rows, annotated = [], []
    for path in images:
        r = yolo.predict(str(path), conf=args.conf, verbose=False)[0]
        im = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB
        im.save(out_dir / f"{path.stem}_pred.jpg", quality=90)
        annotated.append(im)
        for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            rows.append({
                "image_path": str(path), "class_id": int(c), "class_name": r.names[int(c)],
                "confidence": float(s), "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "model": full_name, "model_version": str(version), "inference_ts": ts,
            })
    print(f"{len(images)} images -> {len(rows)} detections")

    if annotated:
        montage(annotated[:12]).save(out_dir / "montage.jpg", quality=90)
        frames = [im.resize((640, int(640 * im.height / im.width))) for im in annotated[:20]]
        frames[0].save(out_dir / "demo.gif", save_all=True, append_images=frames[1:], duration=700, loop=0)
    print(f"Annotated images, montage.jpg, demo.gif -> {out_dir}")

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    table = f"{args.catalog}.{args.schema}.yolo_predictions"
    if rows:
        spark.createDataFrame(rows).write.mode("append").option("mergeSchema", "true").saveAsTable(table)
        spark.sql(f"""SELECT class_name, count(*) AS detections, round(avg(confidence), 3) AS avg_conf
                      FROM {table} WHERE inference_ts = '{ts}' GROUP BY class_name ORDER BY 1""").show()
    print(f"Predictions table: {table}")


if __name__ == "__main__":
    main()
