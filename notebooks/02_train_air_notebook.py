# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · AFTER (minimal migration): the same notebook on `databricks.air`
# MAGIC
# MAGIC **What this is.** `legacy/notebook_before.py` with the smallest change that keeps it working after
# MAGIC `serverless_gpu` is removed: swap the package and rename two decorator arguments. It's still a notebook, still a
# MAGIC decorated function, still one A10. Use it to show that the migration itself is small.
# MAGIC
# MAGIC **Compute.** Serverless GPU, accelerator **1xA10**, environment **AI v6** (`databricks_ai_v6`; torch, ultralytics,
# MAGIC mlflow and `databricks.air` are preinstalled). Single node.
# MAGIC
# MAGIC **Interactive or job?** Both are supported. Attach it to Serverless GPU (1xA10) to run it by hand, or run it as a job
# MAGIC `notebook_task` with `compute.hardware_accelerator: GPU_1xA10` and an environment whose
# MAGIC `base_environment` is `workspace-base-environments/databricks_ai_v6` (see `scripts/run_02_notebook_job.sh`).
# MAGIC The job path requires the admin preview *Serverless workspace base environment support in Jobs*.
# MAGIC Notebook jobs on serverless GPU are capped at 2 days, and you can't add dependencies through the Environments panel.
# MAGIC Verified: as a one-time job run, this notebook starts on an A10 with AI env v6, imports `databricks.air`, and
# MAGIC `@distributed` launches the training process. The single-GPU `RANK` fix below was added after that run
# MAGIC and hasn't been re-run on a GPU yet.
# MAGIC
# MAGIC **This is a pattern demo, not a second champion.** It trains for a few epochs, logs to MLflow, and doesn't
# MAGIC register a model or move `@champion`. The production path is the job's `train` task (`ai_runtime_task`,
# MAGIC generated from `train.yaml`).
# MAGIC
# MAGIC ### Exactly what changed vs `legacy/notebook_before.py`
# MAGIC
# MAGIC | | before | after (this notebook) | why |
# MAGIC |---|---|---|---|
# MAGIC | import | `from serverless_gpu import distributed` | `from databricks.air import distributed` | `serverless_gpu` is deprecated and removed in GPU env v7 |
# MAGIC | GPU count | `gpus=1` | `num_accelerators=1` | renamed argument |
# MAGIC | GPU type | `gpu_type="A10"` | `accelerator_type="GPU_1xA10"` | renamed argument, enum value |
# MAGIC | environment | GPU env v4 + `%pip install ultralytics` | AI env v6, **no `%pip`** | ultralytics is preinstalled. A fresh PyPI torch/opencv stack failed the host's OpenSSL FIPS self-test |
# MAGIC | data | trains on `/Volumes/.../yolo` directly | copies `yolo_dataset.tar` to local disk first | per-file reads through the Volume mount took ~14 min, one tar copy takes seconds |
# MAGIC | MLflow | ultralytics' built-in MLflow callback (implicit) + `log_metric` | callback off, explicit `log_metrics` | the callback manages its own run lifecycle; this keeps everything in the one run `@distributed` creates |
# MAGIC | single-GPU ranks | (not handled) | hide `RANK`/`LOCAL_RANK`/`WORLD_SIZE` while ultralytics imports | `@distributed` exports them for `torch.distributed` but creates no process group. Ultralytics sees `RANK`, builds a `DistributedSampler`, and fails with "Default process group has not been initialized" |
# MAGIC | catalog/schema | hard-coded | widgets (defaults unchanged) | lets a job pass parameters |
# MAGIC | epochs | 30 | 5 (widget) | it's a demo of the pattern |

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "mlb_air_migration", "Schema")
dbutils.widgets.text("epochs", "5", "Epochs")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EPOCHS = int(dbutils.widgets.get("epochs"))
DATASET_TAR = f"/Volumes/{CATALOG}/{SCHEMA}/raw_images/yolo_dataset.tar"

# COMMAND ----------

from databricks.air import distributed  # BEFORE: from serverless_gpu import distributed


@distributed(num_accelerators=1, accelerator_type="GPU_1xA10")  # BEFORE: @distributed(gpus=1, gpu_type="A10")
def train_yolo(epochs: int = 5):
    import os
    import shutil
    import tarfile
    from pathlib import Path

    # One GPU, no torch.distributed process group: hide the rank variables @distributed exports while
    # ultralytics imports (it reads them once, at import), or it builds a DistributedSampler and fails.
    # Restored right after, so @distributed's own per-rank bookkeeping is unaffected.
    saved = {k: os.environ.pop(k) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if k in os.environ}
    try:
        from ultralytics import YOLO, settings
    finally:
        os.environ.update(saved)

    import mlflow
    import yaml

    settings.update({"mlflow": False})  # log explicitly below, into the run @distributed created

    # Volume -> local disk: one sequential tar copy instead of thousands of small-file reads
    work = Path("/tmp/yolo_work")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    shutil.copyfile(DATASET_TAR, work / "yolo_dataset.tar")
    with tarfile.open(work / "yolo_dataset.tar") as tar:
        tar.extractall(work, filter="data")
    data_yaml = work / "yolo" / "data.yaml"
    cfg = yaml.safe_load(data_yaml.read_text())
    cfg["path"] = str(work / "yolo")
    data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

    model = YOLO("yolov8s.pt")
    model.train(data=str(data_yaml), epochs=epochs, imgsz=640, batch=16, device=0,
                project="/tmp/yolo_runs", name="train", exist_ok=True, plots=False)
    m = model.trainer.metrics
    metrics = {"val_mAP50": float(m["metrics/mAP50(B)"]), "val_mAP50_95": float(m["metrics/mAP50-95(B)"])}
    mlflow.log_params({"epochs": epochs, "base_model": "yolov8s.pt", "pattern": "notebook @distributed"})
    mlflow.log_metrics(metrics)
    return metrics


# COMMAND ----------

# Runs on the notebook's attached A10. @distributed creates an MLflow run and prints its link.
# Return values are collected from every rank (one rank here).
result = train_yolo.distributed(epochs=EPOCHS)
metrics = result[0] if isinstance(result, list) else result
print("metrics:", metrics)

# COMMAND ----------

dbutils.notebook.exit(str(metrics))
