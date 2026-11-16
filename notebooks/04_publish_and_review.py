# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Publish & review: predictions → Delta, then look at the results (serverless CPU)
# MAGIC
# MAGIC **What this is.** The last task of the `mlb_air_yolo_train` job (`notebook_task`, serverless CPU), and the
# MAGIC notebook to click through after a run. Interactively, attach it to serverless CPU and **Run all**.
# MAGIC
# MAGIC 1. **Publish:** writes the GPU `infer` task's `predictions.parquet` (latest run) into the Delta table
# MAGIC    `<catalog>.<schema>.yolo_predictions`. Re-running replaces that inference run's rows instead of duplicating them.
# MAGIC 2. **Review:** the `@champion` model's per-class metrics from its MLflow run, the predictions table, and the
# MAGIC    annotated images, montage, and GIF from the artifacts Volume.
# MAGIC
# MAGIC **What changed vs `legacy/notebook_before`.** The legacy notebook ended at one printed `mAP50`, and its results
# MAGIC lived only in the notebook session. Here training and inference are separate GPU job tasks (`ai_runtime_task`).
# MAGIC Their outputs land in Unity Catalog (registered model with `@champion`, Delta table, Volume), so this CPU
# MAGIC notebook can review any run without a GPU attached.

# COMMAND ----------

# MAGIC %pip install --quiet "mlflow-skinny[databricks]>=2.16"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "mlb_air_migration", "Schema")
dbutils.widgets.text("model_name", "mlb_yolo_detector", "Model")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('model_name')}"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/artifacts"
TABLE = f"{CATALOG}.{SCHEMA}.yolo_predictions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Publish the latest GPU inference to Delta

# COMMAND ----------

from pathlib import Path

ts = Path(f"{VOL}/inference/LATEST").read_text().strip()
run_dir = Path(f"{VOL}/inference/{ts}")
df = spark.read.parquet(str(run_dir / "predictions.parquet"))

if spark.catalog.tableExists(TABLE):
    # Idempotent: replace only this inference run's rows
    (df.write.mode("overwrite").option("replaceWhere", f"inference_ts = '{ts}'")
       .option("mergeSchema", "true").saveAsTable(TABLE))
else:
    df.write.saveAsTable(TABLE)
print(f"Published {df.count()} detections from {run_dir} -> {TABLE} (inference_ts={ts})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · The `@champion` model: per-class metrics from its MLflow run

# COMMAND ----------

import mlflow
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias(MODEL, "champion")
run = client.get_run(mv.run_id)
m = run.data.metrics

from databricks.sdk import WorkspaceClient

host = WorkspaceClient().config.host.rstrip("/")
run_url = f"{host}/ml/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
displayHTML(f"""<div style="font-family:sans-serif;font-size:15px">
  <b>{MODEL}</b> version <b>{mv.version}</b> (@champion)<br>
  MLflow run: <a href="{run_url}" target="_blank">{run.info.run_name or run.info.run_id}</a>
  &nbsp;·&nbsp; trained {m.get('train_minutes', float('nan')):.1f} min on 1x A10
  &nbsp;·&nbsp; {run.data.params.get('n_train', '?')} train / {run.data.params.get('n_val', '?')} val images,
  {run.data.params.get('epochs', '?')} epochs
</div>""")

overall = pd.DataFrame([{
    "mAP50": m.get("val_mAP50"), "mAP50-95": m.get("val_mAP50_95"),
    "precision": m.get("val_precision"), "recall": m.get("val_recall")}]).round(3)
per_class = pd.DataFrame(
    [{"class": c, "mAP50-95": m.get(f"val_mAP50_95_{c}")} for c in ["ball", "bat", "glove"]]).round(3)
display(overall)
display(per_class)

# COMMAND ----------

# Training curves and confusion matrix that train.py logged to the run (under plots/)
import base64
import tempfile


def img_tag(path, width="100%"):
    mime = "image/gif" if str(path).endswith(".gif") else "image/png" if str(path).endswith(".png") else "image/jpeg"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:{mime};base64,{data}" style="width:{width};border-radius:6px"/>'


plots = Path(tempfile.mkdtemp())
try:
    mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="plots", dst_path=str(plots))
    wanted = ["results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png"]
    found = [p for w in wanted for p in plots.rglob(w)][:3]
    displayHTML('<div style="display:flex;gap:12px">' +
                "".join(f'<div style="flex:1">{img_tag(p)}<br><small>{p.name}</small></div>' for p in found) +
                "</div>")
except Exception as e:  # plots are a nice-to-have
    print(f"No plots on the champion run: {type(e).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Predictions table

# COMMAND ----------

display(spark.sql(f"""
    SELECT class_name, count(*) AS detections, round(avg(confidence), 3) AS avg_conf,
           count(DISTINCT image_path) AS images
    FROM {TABLE} WHERE inference_ts = '{ts}' GROUP BY class_name ORDER BY class_name"""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT inference_ts, model_version, count(*) AS detections, count(DISTINCT image_path) AS images
    FROM {TABLE} GROUP BY ALL ORDER BY inference_ts DESC"""))

# COMMAND ----------

display(spark.table(TABLE).where(f"inference_ts = '{ts}'")
        .select("image_path", "class_name", "confidence", "x1", "y1", "x2", "y2", "model_version")
        .orderBy("image_path", "class_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · What the model sees: GIF, montage, and annotated test images

# COMMAND ----------

import io

from PIL import Image, ImageSequence

MAX_HTML_BYTES = 3_000_000  # cell output is capped at ~10 MB; base64 adds a third


def small_gif(path, width=400, max_frames=12):
    """Re-encode the GIF smaller if needed, so it renders inline (the original is ~7.5 MB; this is ~2 MB)."""
    raw = Path(path).read_bytes()
    if len(raw) <= MAX_HTML_BYTES:
        return raw
    im = Image.open(path)
    frames = []
    for i, f in enumerate(ImageSequence.Iterator(im)):  # convert while iterating: frames share one object
        if i >= max_frames:
            break
        frames.append(f.convert("RGB").resize((width, int(width * f.height / f.width))))
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=im.info.get("duration", 700), loop=0, optimize=True)
    return buf.getvalue()


gif = run_dir / "demo.gif"
if gif.exists():
    data = base64.b64encode(small_gif(gif)).decode()
    displayHTML(f'<img src="data:image/gif;base64,{data}" style="width:480px;border-radius:6px"/>'
                f"<br><small>demo.gif · inference {ts}</small>")

# COMMAND ----------

montage = run_dir / "montage.jpg"
if montage.exists():
    im = Image.open(montage)
    im.thumbnail((1200, 1200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    data = base64.b64encode(buf.getvalue()).decode()
    displayHTML(f'<img src="data:image/jpeg;base64,{data}" style="width:100%;border-radius:6px"/>'
                "<br><small>montage.jpg · first 12 test images</small>")

# COMMAND ----------

# The most confident detections, one tile per image
top = (spark.table(TABLE).where(f"inference_ts = '{ts}'")
       .groupBy("image_path").agg({"confidence": "max"}).withColumnRenamed("max(confidence)", "conf")
       .orderBy("conf", ascending=False).limit(8).toPandas())
tiles = []
for _, r in top.iterrows():
    pred = run_dir / f"{Path(r.image_path).stem}_pred.jpg"
    if pred.exists():
        tiles.append(f'<div style="width:24%">{img_tag(pred)}<br><small>max conf {r.conf:.2f}</small></div>')
displayHTML('<div style="display:flex;flex-wrap:wrap;gap:8px">' + "".join(tiles) + "</div>")
