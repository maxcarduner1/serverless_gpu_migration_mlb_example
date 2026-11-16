# MLB ball / bat / glove detection on Databricks Serverless GPU (AI Runtime)

This demo fine-tunes **YOLOv8** to detect **ball, bat, and glove** in baseball images. It runs on
**Databricks Serverless GPU (AI Runtime)** with a single **A10** (`GPU_1xA10`).

## Notebooks: before → after

Read and demo these in order:

| Step | File | What it shows | Compute |
|---|---|---|---|
| **before** | `legacy/notebook_before.py` | The old way: a notebook on GPU env v4 with the deprecated `serverless_gpu` `@distributed(gpus=…, gpu_type=…)` decorator. Reference only, not deployed as a task. | Serverless GPU, env v4 |
| **02** | `notebooks/02_train_air_notebook.py` | **Minimal migration.** Same notebook, `serverless_gpu` → `databricks.air`, `gpus`/`gpu_type` → `num_accelerators`/`accelerator_type`. A diff table in its header lists every change. Runs interactively **or** as a job `notebook_task` (`scripts/run_02_notebook_job.sh`). A short pattern demo; it doesn't touch `@champion`. | Serverless GPU 1xA10, AI env v6 |
| **03** | `train.yaml` → job task `train` | **Recommended production path.** `train.yaml` is the single source of truth for the GPU step: iterate with `databricks air run -f train.yaml`, then `databricks air convert-to-dabs` generates the `ai_runtime_task` committed in the job (`scripts/demo_convert_to_dabs.sh` regenerates it and diffs it live). | `ai_runtime_task`, 1xA10 |
| **job** | `resources/mlb_air_yolo_train.job.yml` | The pipeline: **01_prep** notebook → **train** (`ai_runtime_task`) → **infer** (`ai_runtime_task`) → **04_publish_and_review** notebook. | CPU → GPU → GPU → CPU |
| 01 | `notebooks/01_prep.py` | Job task 1: Roboflow → Volume (cached downloads), class merge, subsample, manifest table. Imports `src/prep/prep.py`. | serverless CPU |
| 04 | `notebooks/04_publish_and_review.py` | Job task 4: predictions → Delta, then the champion's per-class metrics, training plots, predictions table, GIF, montage, and annotated images. Click through it after a run. | serverless CPU |

There is no "03 notebook". Calling `databricks air run` from inside a notebook does work, but it starts a
job inside a job with separate retries and run history, needs CLI auth inside the notebook, and loses
lineage in the bundle run. The native `ai_runtime_task` gives you the same GPU step as a first-class task.

**Can a DAB task reference `train.yaml` directly?** No. `ai_runtime_task` only has `code_source_path`,
`command_path`, `deployments[].compute`, `experiment`, `mlflow_run`, and similar fields, and there's no field
that points to a YAML config (`databricks bundle schema`). The task is **generated** from `train.yaml` by
`databricks air convert-to-dabs`. The committed `train` task matches the generated one on compute,
environment, retries/timeout, experiment, and run name. It differs only in the documented bundle-variable
edits (see the comments in the job YAML and the diff printed by `scripts/demo_convert_to_dabs.sh`).

```
databricks.yml                         bundle, variables (catalog/schema/model/experiment), dev target
resources/mlb_air_yolo_train.job.yml   job: 01_prep (notebook) -> train -> infer (ai_runtime_task, GPU_1xA10) -> 04 (notebook)
notebooks/01_prep.py                   job task 1 (serverless CPU): imports src/prep/prep.py
notebooks/02_train_air_notebook.py     AFTER, minimal migration: databricks.air @distributed notebook (interactive or job)
notebooks/04_publish_and_review.py     job task 4 (serverless CPU): predictions -> Delta + metrics/images review
src/prep/prep.py                       prep logic: Roboflow -> UC Volume, class merge, subsample, manifest table
src/train/train.py                     YOLOv8s fine-tune, MLflow logging, Volume checkpoints, UC model registration
src/train/yolo_pyfunc.py               MLflow pyfunc wrapper (image_path -> detections JSON)
src/train/command.sh                   GPU launcher (used by the ai_runtime_task and by train.yaml)
src/infer/infer.py                     GPU batch inference -> annotated images, montage/GIF, predictions.parquet
src/infer/command.sh                   launch script used by the infer ai_runtime_task
train.yaml                             source of truth for the GPU train step (`air run`, `convert-to-dabs`)
scripts/package_code.sh                artifact build: tars src/<component> + run_config.env (bundle vars)
scripts/demo_convert_to_dabs.sh        live demo: convert-to-dabs train.yaml into /tmp, diff vs the committed task
scripts/run_02_notebook_job.sh         one-time job run of notebook 02 on serverless GPU
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

How the variables reach each task: the prep and publish notebooks get `catalog`/`schema`/`model_name` as
`base_parameters` (notebook widgets).
`ai_runtime_task` has no parameters field. So the artifact build (`scripts/package_code.sh`) writes
`run_config.env` (`CATALOG`, `SCHEMA`, `MODEL_NAME`) into the train and infer code tarballs, and
`command.sh` sources it. None of the scripts have catalog/schema defaults. A missing value fails fast.

The job (`max_concurrent_runs: 1`) has four tasks:

1. **prep**: `notebooks/01_prep.py` (`notebook_task`, serverless CPU, `max_retries: 1`)
   * Downloads each Roboflow dataset **once**, as a YOLOv8 export zip, to
     `/Volumes/<catalog>/<schema>/raw_images/sources/`. Later runs reuse the zip and make no Roboflow call.
   * Remaps classes to `0=ball, 1=bat, 2=glove` and subsamples to about 1,800 train / 300 val / 60 test images.
   * Writes `raw_images/yolo/{train,val,test}` + `data.yaml`, a single-file copy `raw_images/yolo_dataset.tar`,
     and the Delta table `<catalog>.<schema>.yolo_manifest`
     (split, per-image class counts, source dataset/version, license, attribution URL).
2. **train** (`ai_runtime_task` generated from `train.yaml`, `GPU_1xA10` × 1, `max_retries: 0`, 1h timeout)
   * Copies `yolo_dataset.tar` (one file, ~150 MB) to local disk, extracts it, and fine-tunes `yolov8s.pt` for 30 epochs at 640px.
   * Logs params, per-epoch metrics (mAP50, mAP50-95, losses), plots, and `results.csv` to MLflow.
   * Mirrors checkpoints to `/Volumes/<catalog>/<schema>/artifacts/checkpoints/<ts>/`.
   * Registers `<catalog>.<schema>.mlb_yolo_detector` in Unity Catalog with the alias `@champion`.
3. **infer** (`ai_runtime_task`, `GPU_1xA10` × 1, same AI v6 environment)
   * Loads `models:/<catalog>.<schema>.mlb_yolo_detector@champion` and runs it on the held-out test split.
   * Writes annotated images, `montage.jpg`, `demo.gif`, and `predictions.parquet` to `artifacts/inference/<ts>/`.
     The montage and GIF are also logged to the MLflow experiment `mlb-air-yolo-infer`.
4. **publish**: `notebooks/04_publish_and_review.py` (`notebook_task`, serverless CPU, `max_retries: 1`)
   * Writes `predictions.parquet` to the Delta table `<catalog>.<schema>.yolo_predictions`. Re-runs replace that
     inference run's rows (`replaceWhere`), so a retry doesn't duplicate rows.
   * Shows the `@champion` model's overall and per-class metrics and training plots from its MLflow run, the
     predictions table, `demo.gif`, `montage.jpg`, and the most confident annotated test images.

How the training code gets to the GPU: the `artifacts.train_code` block in `databricks.yml` packages
`src/train` as a tarball (`type: tgz`). The `ai_runtime_task` gets it via `code_source_path`, unpacks
it on the node, and runs `src/train/command.sh`. This is the same structure that
`databricks air convert-to-dabs train.yaml` generates.

## Path 2: `databricks air run` CLI (dev loop) and `convert-to-dabs`

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
variable defaults. Note that a CLI run also registers a new model version and moves `@champion`, the same as
the job does.

### Live demo: generate the DAB task from `train.yaml`

```bash
bash scripts/demo_convert_to_dabs.sh --profile <profile>     # from the repo root (where train.yaml is)
```

The script copies `train.yaml` and `src/train` to `/tmp/air_dabs_demo` (wiped first) and runs
`databricks air convert-to-dabs train.yaml --output-dir .` there. It prints the generated `databricks.yml`,
`command.sh`, and `env_vars.json`, then a normalized `diff` against the committed `train` task (needs `yq`),
and finally `databricks bundle validate` on the scratch bundle (`--no-validate` to skip). It never writes repo
files and never deploys. `convert-to-dabs` enforces two layout rules:

* `code_source.root_path` must be a **subdirectory** (`src/train`). `.` fails with "resolves to the bundle root".
* `--output-dir` must **contain** the code source. `--output-dir /tmp/x` from the repo fails with
  "not inside the bundle directory", which is why the script converts inside a scratch copy.

`convert-to-dabs` also writes `env_variables` to `generated_artifacts/env_vars.json`, but the generated bundle
never references it (`ai_runtime_task` has no env or parameters field). This repo passes bundle variables
through `run_config.env` in the code tarball instead.

## Before → after (API mapping)

| | before: `serverless_gpu` (GPU env v4) | after: AI Runtime (GPU env v6) |
|---|---|---|
| import | `from serverless_gpu import distributed` | `from databricks.air import distributed` |
| GPU count | `gpus=1` | `num_accelerators=1` / `accelerator_count: 1` |
| GPU type | `gpu_type="A10"` | `accelerator_type="GPU_1xA10"` |
| production launch | notebook job | `ai_runtime_task` in DABs (generated from `train.yaml`), or `databricks air run` |

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
