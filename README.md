# MLB ball / bat / glove detection on Databricks Serverless GPU (AI Runtime)

This demo fine-tunes **YOLOv8** to detect **ball, bat, and glove** in baseball images. It runs on
**Databricks Serverless GPU (AI Runtime)** with a single **A10** (`GPU_1xA10`). The same
training step can be launched two ways:

| Path | When to use | Entry point |
|---|---|---|
| **DABs `ai_runtime_task`** | scheduled / CI-CD-deployed pipelines (CPU prep → GPU train → GPU infer → CPU publish) | `databricks.yml` + `resources/mlb_air_yolo_train.job.yml` |
| **`databricks air run` CLI** | fast dev loop on the training script | `train.yaml` |

`legacy/notebook_before.py` shows the **old** way: a notebook on GPU env v4 with the deprecated
`serverless_gpu` `@distributed(gpus=…, gpu_type=…)` decorator. It is reference only and is not
deployed. See `docs/migration.md` for the full migration guide.

```
databricks.yml                         bundle, variables (catalog/schema/model/experiment), dev target
resources/mlb_air_yolo_train.job.yml   job: prep -> train -> infer (both ai_runtime_task, GPU_1xA10) -> publish
src/prep/prep.py                       serverless CPU: Roboflow -> UC Volume, class merge, subsample, manifest table
src/train/train.py                     YOLOv8s fine-tune, MLflow logging, Volume checkpoints, UC model registration
src/train/yolo_pyfunc.py               MLflow pyfunc wrapper (image_path -> detections JSON)
src/train/command.sh                   GPU launcher (used by the ai_runtime_task and by train.yaml)
src/infer/infer.py                     GPU batch inference -> annotated images, montage/GIF, predictions.parquet
src/infer/command.sh                   launch script used by the infer ai_runtime_task
src/publish/publish.py                 serverless CPU: predictions.parquet -> Delta table
train.yaml                             `databricks air run` workload (same training code and launcher)
scripts/package_code.sh                artifact build: tars src/<component> + run_config.env (bundle vars)
legacy/notebook_before.py              BEFORE: serverless_gpu notebook (deprecated API)
docs/migration.md                      serverless_gpu -> AI Runtime migration notes
```

## Prerequisites

* Databricks CLI ≥ 1.22 (`databricks air`, `ai_runtime_task` in bundles), plus a CLI profile for your
  workspace (`databricks auth login --host <workspace-url> --profile <profile>`). The bundle
  doesn't hardcode a host. The workspace comes from `-p <profile>` (or `DATABRICKS_HOST`).
* `bash` and `tar` locally (the artifact build step packages the GPU code).
* A Unity Catalog schema. Defaults are `cmegdemos_catalog.mlb_air_migration`; override them with
  `--var catalog=… --var schema=…` (and `--var model_name=…`) on validate/deploy/run.
* Two volumes in that schema:
  ```bash
  databricks volumes create cmegdemos_catalog mlb_air_migration raw_images MANAGED -p <profile>
  databricks volumes create cmegdemos_catalog mlb_air_migration artifacts  MANAGED -p <profile>
  ```
* A Roboflow API key in a secret scope. Only the CPU prep task reads it. The GPU task needs no secrets.
  ```bash
  databricks secrets create-scope roboflow -p <profile>
  databricks secrets put-secret roboflow roboflow_api_key -p <profile>
  ```

## Path 1: bundle with `ai_runtime_task`

```bash
databricks bundle validate -t dev -p <profile>
databricks bundle deploy   -t dev -p <profile>
databricks bundle run mlb_air_yolo_train -t dev -p <profile>

# another catalog/schema: pass the same --var flags to deploy (they are baked into the GPU code archives)
databricks bundle deploy -t dev -p <profile> --var catalog=my_catalog --var schema=my_schema
```

How the variables reach each task: prep and publish get `--catalog/--schema` as task parameters.
`ai_runtime_task` has no parameters field. So the artifact build (`scripts/package_code.sh`) writes
`run_config.env` (`CATALOG`, `SCHEMA`, `MODEL_NAME`) into the train and infer code tarballs, and
`command.sh` sources it. None of the scripts have catalog/schema defaults. A missing value fails fast.

The job (`max_concurrent_runs: 1`) has four tasks:

1. **prep** (serverless CPU, `max_retries: 1`)
   * Downloads each Roboflow dataset **once**, as a YOLOv8 export zip, to
     `/Volumes/<catalog>/<schema>/raw_images/sources/`. Later runs reuse the zip and make no Roboflow call.
   * Remaps classes to `0=ball, 1=bat, 2=glove` and subsamples to about 1,800 train / 300 val / 60 test images.
   * Writes `raw_images/yolo/{train,val,test}` + `data.yaml`, a single-file copy `raw_images/yolo_dataset.tar`,
     and the Delta table `<catalog>.<schema>.yolo_manifest`
     (split, per-image class counts, source dataset/version, license, attribution URL).
2. **train** (`ai_runtime_task`, `GPU_1xA10` × 1, `max_retries: 0`, 1h timeout)
   * Copies `yolo_dataset.tar` (one file, ~150 MB) to local disk, extracts it, and fine-tunes `yolov8s.pt` for 30 epochs at 640px.
   * Logs params, per-epoch metrics (mAP50, mAP50-95, losses), plots, and `results.csv` to MLflow.
   * Mirrors checkpoints to `/Volumes/<catalog>/<schema>/artifacts/checkpoints/<ts>/`.
   * Registers `<catalog>.<schema>.mlb_yolo_detector` in Unity Catalog with the alias `@champion`.
3. **infer** (`ai_runtime_task`, `GPU_1xA10` × 1, same AI v6 environment)
   * Loads `models:/<catalog>.<schema>.mlb_yolo_detector@champion` and runs it on the held-out test split.
   * Writes annotated images, `montage.jpg`, `demo.gif`, and `predictions.parquet` to `artifacts/inference/<ts>/`.
     The montage and GIF are also logged to the MLflow experiment `mlb-air-yolo-infer`.
4. **publish** (serverless CPU)
   * Appends `predictions.parquet` to the Delta table `<catalog>.<schema>.yolo_predictions`.

How the training code gets to the GPU: the `artifacts.train_code` block in `databricks.yml` packages
`src/train` as a tarball (`type: tgz`). The `ai_runtime_task` gets it via `code_source_path`, unpacks
it on the node, and runs `src/train/command.sh`. This is the same structure that
`databricks air convert-to-dabs train.yaml` generates.

## Path 2: `databricks air run` CLI

Once prep has run at least once (the dataset is on the Volume):

```bash
databricks air run -f train.yaml --dry-run -p <profile>   # validate against the workspace
databricks air run -f train.yaml --watch   -p <profile>   # submit + stream logs
databricks air run -f train.yaml --watch   -p <profile> \
    --override env_variables.CATALOG=my_catalog --override env_variables.SCHEMA=my_schema
```

`train.yaml` uploads only `src/train` (`code_source.root_path`). It uses the same accelerator
(`GPU_1xA10`) and the same AI v6 environment (`version: databricks_ai_v6`) as the bundle task, with
`max_retries: 0` and `timeout_minutes: 60`. It runs the same `src/train/command.sh`. Catalog, schema,
and model come from `env_variables` (`CATALOG`, `SCHEMA`, `MODEL_NAME`), which mirror the bundle
variable defaults. To turn it into a bundle scaffold:

```bash
databricks air convert-to-dabs train.yaml --output-dir /tmp/air-bundle
```

## Before → after (API mapping)

| | before: `serverless_gpu` (GPU env v4) | after: AI Runtime (GPU env v6) |
|---|---|---|
| import | `from serverless_gpu import distributed` | `from databricks.air import distributed` |
| GPU count | `gpus=1` | `num_accelerators=1` / `accelerator_count: 1` |
| GPU type | `gpu_type="A10"` | `accelerator_type="GPU_1xA10"` |
| production launch | notebook job | `ai_runtime_task` in DABs, or `databricks air run` |

## Datasets, licenses, and attribution

All data comes from public [Roboflow Universe](https://universe.roboflow.com) projects. Each image's
source and license is recorded in the `yolo_manifest` table. Downloads are capped (one export per
dataset, cached in the Volume) and subsampled, to stay well inside Roboflow free-tier limits.

| Dataset | Classes used | License | Attribution |
|---|---|---|---|
| `kschattin/pitch-tracking` | `baseball` → ball | MIT | https://universe.roboflow.com/kschattin/pitch-tracking |
| `baseball-94erg/baseball-detection-v1-vvupe` | `Ball` → ball, `Bat` → bat (class `0` dropped) | CC BY 4.0 | https://universe.roboflow.com/baseball-94erg/baseball-detection-v1-vvupe |
| `tel-4fuev/baseball-glove-j2hci` | `glove` → glove | CC BY 4.0 | https://universe.roboflow.com/tel-4fuev/baseball-glove-j2hci |

CC BY 4.0 requires attribution. Credit the dataset authors above when you share results or images.
Pretrained weights (`yolov8s.pt`) and the `ultralytics` library are © Ultralytics, under AGPL-3.0.

## Notes and lessons learned

* GPU: this demo uses **`GPU_1xA10` only**. Change `accelerator_type` / `accelerator_count` in the job
  YAML or `train.yaml` to scale up.
* **Use the AI environment for GPU tasks.** `environment.version: databricks_ai_v6` in `train.yaml`
  maps to `base_environment: workspace-base-environments/databricks_ai_v6` in DABs (that's what
  `databricks air convert-to-dabs` emits). It ships torch 2.11 (CUDA 13), ultralytics 8.4.90, and mlflow.
  The Standard v6 image has no torch. Pip-installing `ultralytics` on top pulled the newest torch,
  opencv, and CUDA wheels from PyPI, and the process aborted with `OpenSSL ... FATAL FIPS SELFTEST FAILURE`.
  The same thing happened for CPU inference on serverless with PyPI ultralytics, which is why
  inference also runs on the AI GPU env and only the Delta write runs on CPU.
* **Ship many small files as one archive.** Copying about 4.3k images and labels file-by-file from
  the Volume took about 14 minutes on the GPU node. Copying one ~150 MB tar and extracting it locally took 3 s.
* `ai_runtime_task.code_source_path` is resolved from the bundle root, not relative to the
  resources file. Inside the node, the archive is unpacked at `$CODE_SOURCE_PATH/<subdir>`, and
  `MLFLOW_RUN_ID` / `MLFLOW_EXPERIMENT_ID` are injected (`mlflow.start_run()` resumes that run).
* `ai_runtime_task` has no task parameters. Catalog, schema, and model reach the GPU nodes via
  `run_config.env` in the code tarball (see above). Epochs and base model are `EPOCHS` / `BASE_MODEL`
  in `command.sh` (defaults 30 / `yolov8s.pt`).
* The Roboflow key is read only in the prep task. All Roboflow HTTP errors are re-raised with the
  query string (API key, signed download URL) removed. The signed export link is never printed.
* Dataset size: `prep.py --max_train/--max_val/--max_test`. Training length: `--epochs`. 30 epochs on
  1,800 images took about 8.5 min of training on one A10 (about 11 min for the whole task, including
  provisioning and environment setup).
