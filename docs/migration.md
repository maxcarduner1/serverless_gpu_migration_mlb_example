# Migrating from `serverless_gpu` notebooks to the AI Runtime CLI and DABs

Researched 2026-11-16 against Databricks CLI v1.22.1 and the public docs.

**Confidence labels:** **H** = confirmed in docs or CLI output. **M** = inferred, or the docs are ambiguous. **L** = unverified.

---

## TL;DR

| Question | Answer |
|---|---|
| Can GPU training be a DAB job task? | **Yes.** Use the native `ai_runtime_task` task type (Public Preview, H). |
| Prep task, then GPU task, in one job? | **Yes.** Use an ordinary `depends_on` DAG. CPU prep runs on serverless; GPU training runs as `ai_runtime_task`. This is the documented pattern (H). |
| Should a notebook task call `databricks air run`? | **No.** It creates a hidden nested run, wastes idle CPU while it polls, splits retries and lineage, and needs CLI auth inside the job. |
| Can a DAB `ai_runtime_task` reference a `train.yaml`? | **No, it has to be generated.** The task schema has no field that points to a YAML config. `databricks air convert-to-dabs train.yaml` generates the task, so keep `train.yaml` as the source of truth and diff the committed task against a fresh conversion (H, CLI v1.22.1 schema + output). |
| Can a `@distributed` notebook on serverless GPU run as a job task? | **Yes.** Use a `notebook_task` with `compute.hardware_accelerator` and an AI base environment. See §1A. |
| Migration effort | **S** for the required package rename. **M** to move from notebook to CLI script, most of which is re-validating dependencies on CUDA 13 and torch 2.11. |

---

## 1. Two separate moves

### A. Package swap (required before GPU env v7). Size: S

```python
# BEFORE (env v4)
from serverless_gpu import distributed

@distributed(gpus=8, gpu_type="H100")
def train(): ...

train.distributed()
```

```python
# AFTER (env v6)
from databricks.air import distributed   # pip: databricks-sdk-air

@distributed(num_accelerators=8, accelerator_type="GPU_8xH100")
def train(): ...
```

- `@distributed` is **Beta and single-node**. Its default timeout is 3 h; change it with `timeout=` (seconds) or set `timeout=None` (H).
- Notebook sessions and notebook jobs are capped at **2 days**.
- Don't `%pip install` the GPU stack on the AI environment. torch, ultralytics, mlflow and `databricks.air` are preinstalled in `databricks_ai_v6`, and a fresh PyPI torch/opencv stack can fail the host's OpenSSL FIPS self-test.

**The same notebook as a job task (no code change):**

```yaml
tasks:
  - task_key: train_notebook
    notebook_task: {notebook_path: ../notebooks/02_train_air_notebook.py}
    compute: {hardware_accelerator: GPU_1xA10}          # selects Serverless GPU for the notebook
    environment_key: ai_v6
environments:
  - environment_key: ai_v6
    spec: {base_environment: workspace-base-environments/databricks_ai_v6}
```

A workspace admin must enable the *Serverless workspace base environment support in Jobs* preview. You can't add
dependencies through the Environments panel for these jobs. Use `%pip` in the notebook for anything extra.

### B. Notebook to CLI script (optional, recommended for production). Size: M

| Area | Notebook with `@distributed` | CLI / `ai_runtime_task` |
|---|---|---|
| Entry point | Decorated function in a cell | `src/train.py` with `main()` |
| Launch | The decorator spawns ranks | You run `torchrun` yourself, using the injected `NUM_NODES`, `NODE_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` |
| Dependencies | `%pip` | `environment.dependencies` in the YAML |
| Data | UC Volumes and tables | Same. The run-as identity needs Volume grants. Read from Volumes; don't assume a Spark session |
| MLflow | Run created automatically | Experiment created per run; log from rank 0 |
| Secrets | `dbutils.secrets` | CLI YAML `secrets: {NAME: scope/key}`. **Unconfirmed in DABs** (see §4) |
| Max runtime | 2 days | 14 days |
| Multi-node | No | Yes, with `GPU_8xH100` in multiples of 8 |

```yaml
# train.yaml -> databricks air run -f train.yaml --watch
experiment_name: mlb-air-yolo
compute: {accelerator_type: GPU_1xA10, num_accelerators: 1}
environment: {version: 6, dependencies: ["databricks-sdk-air[data]"]}
code_source: {root_path: src}      # must be a subdirectory, NOT "." (convert-to-dabs rejects ".")
max_retries: 0                     # the CLI default is 3; set it explicitly
timeout_minutes: 60
command: torchrun --nnodes=$NUM_NODES --node_rank=$NODE_RANK --nproc_per_node=$LOCAL_WORLD_SIZE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $CODE_SOURCE_PATH/train/train.py
```

**Useful CLI commands:**
- `databricks air run|get|list|logs|cancel|images|pools|convert-to-dabs`
- `databricks air run -h config.<field>` shows the documentation for any YAML field.

---

## 2. Recommended workflow for a team that runs everything through DABs

The story, notebook-first:

1. **Before:** a `serverless_gpu` notebook on GPU env v4.
2. **Minimal migration:** the same notebook on `databricks.air` (package swap + argument renames). It runs
   interactively or as a job `notebook_task` on serverless GPU (§1A).
3. **Production GPU step:** move the training code into a script with a `train.yaml`. Iterate with
   `databricks air run -f train.yaml --watch`, then generate the job task with `databricks air convert-to-dabs`.
4. **Job:** CPU notebook tasks around the native `ai_runtime_task`(s): prep notebook → train → infer → publish/review notebook.
5. **Promote** from dev to staging to prod through the existing CI/CD.

Don't add a notebook that shells out to `databricks air run`. It works, but it adds a job inside a job, separate
retries and run history, CLI install and auth inside the notebook, and weaker lineage in the bundle run.

### `train.yaml` is the source of truth: generate, don't hand-write

An `ai_runtime_task` can't reference `train.yaml`. Generate it, then merge it into the bundle:

```bash
# convert-to-dabs requires: root_path is a subdirectory (not "."), and --output-dir contains the code source
cp -R train.yaml src /tmp/air_dabs_demo/ && cd /tmp/air_dabs_demo
databricks air convert-to-dabs train.yaml --output-dir .
```

What it generates and what you usually change when merging:

| Generated | Typical edit when merging into an existing bundle |
|---|---|
| `artifacts.code_source` (tgz of `root_path`) → `code_source_path: ./dist/code_source.tgz` | Keep the same layout. Add a build step if the task needs config baked in |
| `deployments[0].command_path: ./generated_artifacts/command.sh` (= `train.yaml` `command`) | Point at your own launcher if the generated one only wraps it |
| `generated_artifacts/env_vars.json` (= `env_variables`) | **Not referenced by the generated bundle.** The task has no env or parameters field, so bake values into the code archive (e.g. a `run_config.env`) or read them from a Volume |
| `experiment`, `mlflow_run`, `max_retries`, `timeout_seconds`, `compute`, `base_environment` | Keep. Swap literals for bundle variables if needed |

Re-run the conversion in CI or before a demo and diff it against the committed task, so the two can't drift.

### The DAB job shape (edited output of `convert-to-dabs`)

```yaml
resources:
  jobs:
    mlb_air_yolo_train:
      max_concurrent_runs: 1            # docs recommend this when tasks hand off via a Volume
      tasks:
        - task_key: prep                # notebook on serverless CPU
          max_retries: 1
          notebook_task: {notebook_path: ../notebooks/01_prep.py, base_parameters: {catalog: ..., schema: ...}}
        - task_key: train               # generated from train.yaml
          depends_on: [{task_key: prep}]
          environment_key: train_env
          max_retries: 0
          timeout_seconds: 3600
          ai_runtime_task:
            experiment: mlb-air-yolo
            code_source_path: ./dist/train_code.tgz    # an artifacts: tgz build; resolved from the bundle root
            deployments:                               # only one entry allowed in Preview
              - command_path: ../src/train/command.sh
                compute: {accelerator_type: GPU_1xA10, accelerator_count: 1}
        - task_key: infer               # ai_runtime_task, same shape
          depends_on: [{task_key: train}]
        - task_key: publish             # notebook on serverless CPU: Delta write + review
          depends_on: [{task_key: infer}]
          notebook_task: {notebook_path: ../notebooks/04_publish_and_review.py}
      environments:
        - environment_key: train_env
          spec: {base_environment: workspace-base-environments/databricks_ai_v6}
```

See the demo bundle in this repo for the working version.

---

## 3. Options compared

| Option | Verdict |
|---|---|
| **(a) Native `ai_runtime_task`** | **Recommended.** Declarative, in git, multi-node, 14-day cap, job-level retries, and a normal DAG. Downsides: Preview, one deployment only, and the secrets gap. |
| (c) `notebook_task` with `compute.hardware_accelerator` and `@distributed` | Smallest change for a notebook-first team, and it works as a job task (verified, §1A). Single-node, 2-day cap. The AI base env in Jobs is Beta and preview-gated. |
| (b) A task that shells out to `air run`, the SDK or REST | **Avoid.** Hidden nested run, idle CPU while polling, retries and lineage split across two runs. |
| (d) `run_job_task` pointing at a separate GPU job | Only worth it to reuse one training job across several parent pipelines. |
| (d) `python_wheel_task` / `spark_python_task` with a GPU accelerator | Allowed by the schema, but there are no docs for it (L). Don't recommend it. |

---

## 4. Gotchas and open questions

- **Preview status (H):** `databricks air` and `ai_runtime_task` are both Public Preview. `gen_ai_compute_task` is **deprecated**; flag any bundle that still uses it.
- **`root_path: .` (H):** `convert-to-dabs` rejects it. Keep code under `src/`.
- **Env vars and secrets in `ai_runtime_task` (H for env vars, M–L for secrets):** the task schema has no env, parameters, or secrets field. `convert-to-dabs` emits `env_vars.json` (and `secret_env_vars.json` for secrets), but the generated bundle never references them. Fetch secrets in the CPU task, or test before relying on them.
- **Multi-node (M):** the overview page says "single node"; the CLI reference documents 16 GPUs as 2 nodes. No maximum node count is published.
- **Retries (H):** `air run` defaults to 3 retries, and `timeout_minutes` covers all of them. Set both explicitly.
- **Capacity (H):** on-demand only. Under demand, cross-region GPUs may be used, which can incur egress. Reserved capacity (`provisioned_capacity_id`, `priority_class`) is Private Preview; ask your account team about it.
- **Hardware (H):** A10 and 1×H100 are single-GPU only. Multi-GPU needs 8×H100 or 8×B300, and B300 is Preview with a waitlist.
- **Compliance security profile (H):** an admin must enable the "AI Runtime" and "AI Runtime multi-node H100" previews. FedRAMP High and IL5 aren't supported.
- **Custom images (H):** `unity_catalog_image` needs the Artifact Registry preview and is mutually exclusive with `dependencies` and `version`.
- **Lifecycle (H):** `bundle deploy` creates a persistent job, whereas `air run` creates ephemeral runs.

**Still open (ask your Databricks account team):**
- When does env v7 ship?
- Do secrets work in `ai_runtime_task`?
- What's the maximum node count?
- When do `air` and `ai_runtime_task` become GA?
