#!/bin/bash
# Run notebooks/02_train_air_notebook ONCE as a one-time job run (notebook_task on Serverless GPU, 1x A10).
# It doesn't create a persistent job. Deploy the bundle first so the notebook exists in the workspace.
#
#   bash scripts/run_02_notebook_job.sh [--profile DEFAULT] [--epochs 5] [--target dev] \
#       [--catalog <c>] [--schema <s>]
#
# catalog/schema default to the bundle variables of the chosen target (`databricks bundle summary`),
# so the run never falls back silently to the notebook's widget defaults. (Notebook 02 registers no
# model, so it takes no model_name.)
#
# The task shape that makes a notebook run on serverless GPU as a job task:
#   compute.hardware_accelerator: GPU_1xA10
#   environment_key -> spec.base_environment: workspace-base-environments/databricks_ai_v6
# (Requires the workspace preview "Serverless workspace base environment support in Jobs".)
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE=DEFAULT EPOCHS=5 TARGET=dev CATALOG="" SCHEMA=""
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--profile) PROFILE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    -t|--target) TARGET="$2"; shift 2 ;;
    --catalog) CATALOG="$2"; shift 2 ;;
    --schema) SCHEMA="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$EPOCHS" =~ ^[0-9]+$ ]] || { echo "--epochs must be an integer" >&2; exit 2; }

SUMMARY=$(databricks bundle summary -t "$TARGET" -p "$PROFILE" -o json)
read -r FILES BUNDLE_CATALOG BUNDLE_SCHEMA < <(printf '%s' "$SUMMARY" | python3 -c '
import json, sys
s = json.load(sys.stdin)
v = s.get("variables", {})
print(s["workspace"]["file_path"], v["catalog"]["value"], v["schema"]["value"])')
CATALOG="${CATALOG:-$BUNDLE_CATALOG}" SCHEMA="${SCHEMA:-$BUNDLE_SCHEMA}"
for x in "$CATALOG" "$SCHEMA"; do
  [[ "$x" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid catalog/schema: '$x'" >&2; exit 2; }
done
NB="$FILES/notebooks/02_train_air_notebook"
echo "notebook=$NB catalog=$CATALOG schema=$SCHEMA epochs=$EPOCHS"

REQ="$(mktemp -t run02.XXXXXX)"
trap 'rm -f "$REQ"' EXIT
cat > "$REQ" <<EOF
{
  "run_name": "02_train_air_notebook (one-time, notebook on serverless GPU)",
  "timeout_seconds": 3600,
  "tasks": [{
    "task_key": "train_notebook",
    "notebook_task": {"notebook_path": "$NB", "base_parameters": {"catalog": "$CATALOG", "schema": "$SCHEMA", "epochs": "$EPOCHS"}},
    "compute": {"hardware_accelerator": "GPU_1xA10"},
    "environment_key": "ai_v6",
    "max_retries": 0,
    "timeout_seconds": 3600
  }],
  "environments": [{
    "environment_key": "ai_v6",
    "spec": {"base_environment": "workspace-base-environments/databricks_ai_v6"}
  }]
}
EOF

# --no-wait: print the run id right away; follow it in the Jobs UI or with `databricks jobs get-run <id>`
RUN_ID=$(databricks jobs submit -p "$PROFILE" --no-wait --json "@$REQ" -o json | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["run_id"])')
echo "submitted one-time run: $RUN_ID"
databricks jobs get-run "$RUN_ID" -p "$PROFILE" -o json | python3 -c \
  'import json,sys; print("run page:", json.load(sys.stdin)["run_page_url"])'
