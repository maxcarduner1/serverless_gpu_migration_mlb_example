# Databricks notebook source
# MAGIC %md
# MAGIC # BEFORE: YOLOv8 training with the deprecated `serverless_gpu` package (GPU env v4)
# MAGIC
# MAGIC > **This is the legacy "before" version, for reference only.** It is **not** part of the bundle job.
# MAGIC > The "after" version is `src/train/train.py`, launched by the `ai_runtime_task` in
# MAGIC > `resources/mlb_air_yolo_train.job.yml` or by `databricks air run -f train.yaml`.
# MAGIC
# MAGIC Pattern: a notebook attached to **Serverless GPU, environment v4**, using
# MAGIC `serverless_gpu.distributed` with the old argument names `gpus=` / `gpu_type="A10"`.
# MAGIC
# MAGIC | | before (this notebook) | after (this repo) |
# MAGIC |---|---|---|
# MAGIC | Package | `serverless_gpu` (deprecated; dropped in env v7) | `databricks.air` / `databricks air` CLI / `ai_runtime_task` |
# MAGIC | GPU args | `gpus=1, gpu_type="A10"` | `num_accelerators=1, accelerator_type="GPU_1xA10"` |
# MAGIC | Env | GPU env v4 (CUDA 12, torch 2.7) | GPU env v6 (CUDA 13, torch 2.11) |
# MAGIC | Lifecycle | tied to the notebook session | Jobs run + MLflow run, deployed by DABs CI/CD |

# COMMAND ----------

# MAGIC %pip install ultralytics==8.3.* --quiet
# MAGIC %restart_python

# COMMAND ----------

from serverless_gpu import distributed  # DEPRECATED. After: from databricks.air import distributed

CATALOG, SCHEMA = "cmegdemos_catalog", "mlb_air_migration"
DATA_YAML = f"/Volumes/{CATALOG}/{SCHEMA}/raw_images/yolo/data.yaml"


@distributed(gpus=1, gpu_type="A10")  # After: @distributed(num_accelerators=1, accelerator_type="GPU_1xA10")
def train_yolo(epochs: int = 30):
    import mlflow
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    results = model.train(data=DATA_YAML, epochs=epochs, imgsz=640, batch=16, device=0,
                          project="/tmp/yolo_runs", name="train")
    mlflow.log_metric("val_mAP50", float(results.box.map50))
    return float(results.box.map50)


# COMMAND ----------

# Runs remotely on the GPU. The run ends when this notebook session ends.
map50 = train_yolo.distributed(epochs=30)
print("mAP50:", map50)
