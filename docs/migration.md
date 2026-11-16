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

1. **Explore** in notebooks on env v6 with `databricks.air`.
2. **Iterate** on the script with `databricks air run -f train.yaml --watch`.
3. **Scaffold** the task once with `databricks air convert-to-dabs`.
4. **Merge** the generated `ai_runtime_task` and its artifacts into the existing bundle, after the CPU prep tasks.
5. **Promote** from dev to staging to prod through the existing CI/CD.

### The DAB job shape (edited output of `convert-to-dabs`)

```yaml
resources:
  jobs:
    mlb_air_yolo_train:
      max_concurrent_runs: 1            # docs recommend this when tasks hand off via a Volume
      tasks:
        - task_key: prep                # serverless CPU
          notebook_task: {notebook_path: ../src/prep/prep.py}
        - task_key: train
          depends_on: [{task_key: prep}]
          max_retries: 0
          timeout_seconds: 3600
          ai_runtime_task:
            experiment: mlb-air-yolo
            code_source_path: ./dist/code_source.tgz   # filled in by tooling; don't write it by hand
            deployments:                               # only one entry allowed in Preview
              - command_path: ./generated_artifacts/command.sh
                compute: {accelerator_type: GPU_1xA10, accelerator_count: 1}
        - task_key: infer
          depends_on: [{task_key: train}]
```

See the demo bundle in this repo for the working version.

---

## 3. Options compared

| Option | Verdict |
|---|---|
| **(a) Native `ai_runtime_task`** | **Recommended.** Declarative, in git, multi-node, 14-day cap, job-level retries, and a normal DAG. Downsides: Preview, one deployment only, and the secrets gap. |
| (c) `notebook_task` with `compute.hardware_accelerator` and `@distributed` | Smallest change for a notebook-first team. Single-node, 2-day cap. The AI base env in Jobs is Beta and preview-gated. |
| (b) A task that shells out to `air run`, the SDK or REST | **Avoid.** Hidden nested run, idle CPU while polling, retries and lineage split across two runs. |
| (d) `run_job_task` pointing at a separate GPU job | Only worth it to reuse one training job across several parent pipelines. |
| (d) `python_wheel_task` / `spark_python_task` with a GPU accelerator | Allowed by the schema, but there are no docs for it (L). Don't recommend it. |

---

## 4. Gotchas and open questions

- **Preview status (H):** `databricks air` and `ai_runtime_task` are both Public Preview. `gen_ai_compute_task` is **deprecated**; flag any bundle that still uses it.
- **`root_path: .` (H):** `convert-to-dabs` rejects it. Keep code under `src/`.
- **Secrets in `ai_runtime_task` (M–L):** the task schema has no secrets field. `convert-to-dabs` emits `secret_env_vars.json`, but the bundle never wires it in. Fetch secrets in the CPU task, or test before relying on them.
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
