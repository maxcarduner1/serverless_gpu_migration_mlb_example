# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Prep: Roboflow → UC Volume → YOLO dataset + manifest table (serverless CPU)
# MAGIC
# MAGIC **What this is.** The first task of the `mlb_air_yolo_train` job (`notebook_task`, serverless CPU).
# MAGIC You can also run it interactively on serverless CPU.
# MAGIC
# MAGIC 1. Downloads each public Roboflow Universe dataset **once**, as a YOLOv8 export zip, to
# MAGIC    `/Volumes/<catalog>/<schema>/raw_images/sources/`. Later runs reuse the cached zips and make no Roboflow call.
# MAGIC 2. Merges classes into `0=ball, 1=bat, 2=glove` and subsamples (~1,800 train / 300 val / 60 test).
# MAGIC 3. Writes `raw_images/yolo/` + `data.yaml`, a single-file `raw_images/yolo_dataset.tar` for fast GPU-side copy,
# MAGIC    and the Delta table `<catalog>.<schema>.yolo_manifest`.
# MAGIC
# MAGIC **What changed vs `legacy/notebook_before`.** The legacy notebook assumed the dataset already sat in the Volume
# MAGIC and read it file by file from the GPU. Here data prep is its own **CPU** step, so no GPU time is spent on
# MAGIC downloads or class remapping. The only secret (the Roboflow key, `roboflow/roboflow_api_key`) is read here
# MAGIC and never on the GPU.
# MAGIC
# MAGIC The logic lives in `src/prep/prep.py` and is imported, not copied, so the notebook and the module can't drift.

# COMMAND ----------

# MAGIC %pip install --quiet pyyaml requests
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "cmegdemos_catalog", "Catalog")
dbutils.widgets.text("schema", "mlb_air_migration", "Schema")
dbutils.widgets.text("model_name", "mlb_yolo_detector", "Model")  # not used by prep; every job task gets the same parameters
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
print(f"catalog={CATALOG} schema={SCHEMA} model_name={dbutils.widgets.get('model_name')}")

# COMMAND ----------

# Import the shared prep module from the bundle (notebooks/ and src/ are deployed side by side).
import os
import sys

sys.path.insert(0, os.path.abspath("../src/prep"))
import prep  # noqa: E402

manifest_table = prep.main(["--catalog", CATALOG, "--schema", SCHEMA])

# COMMAND ----------

# MAGIC %md
# MAGIC ### What landed in Unity Catalog

# COMMAND ----------

display(spark.sql(f"""
    SELECT split, source_dataset, license, count(*) AS images,
           sum(n_ball) AS ball, sum(n_bat) AS bat, sum(n_glove) AS glove
    FROM {manifest_table} GROUP BY ALL ORDER BY split, source_dataset"""))

# COMMAND ----------

vol = f"/Volumes/{CATALOG}/{SCHEMA}/raw_images"
print(open(f"{vol}/yolo/data.yaml").read())
print(f"yolo_dataset.tar: {os.path.getsize(f'{vol}/yolo_dataset.tar') / 1e6:.1f} MB")
