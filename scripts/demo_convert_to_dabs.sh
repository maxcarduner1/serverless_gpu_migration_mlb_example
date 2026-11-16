#!/bin/bash
# Live demo: generate a DAB ai_runtime_task from train.yaml, then diff it against the committed
# `train` task in resources/mlb_air_yolo_train.job.yml.
#
#   cd <repo root>                       # the directory that contains train.yaml
#   bash scripts/demo_convert_to_dabs.sh [--profile DEFAULT] [--out /tmp/air_dabs_demo] [--no-validate]
#
# Safe to run on a call:
#   * writes ONLY to the scratch dir (wiped first); never touches repo files
#   * never deploys or runs anything. With a profile it runs `databricks bundle validate` on the
#     generated scratch bundle (read-only). Pass --no-validate to stay fully offline.
#
# Layout rules that convert-to-dabs enforces (both checked live):
#   * code_source.root_path must be a SUBDIRECTORY (src/train), not "." ("resolves to the bundle root").
#   * --output-dir must contain the code_source ("not inside the bundle directory"), so this script
#     copies train.yaml and its root_path into the scratch dir and converts there.
set -euo pipefail

PROFILE=DEFAULT
OUT=/tmp/air_dabs_demo
VALIDATE=1
JOB_FILE=resources/mlb_air_yolo_train.job.yml
while [ $# -gt 0 ]; do
  case "$1" in
    -p|--profile) PROFILE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --no-validate) VALIDATE=0; shift ;;
    --job-file) JOB_FILE="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -f train.yaml ] || { echo "run this from the directory that contains train.yaml" >&2; exit 1; }
REPO="$(pwd -P)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd -P)"
case "$OUT/" in
  "$REPO"/*|/) echo "refusing to use $OUT as scratch dir (inside the repo or /)" >&2; exit 1 ;;
esac

ROOT_PATH="$(sed -n 's/^[[:space:]]*root_path:[[:space:]]*\([^[:space:]#]*\).*/\1/p' train.yaml | head -1)"
[ -n "$ROOT_PATH" ] && [ -d "$ROOT_PATH" ] || { echo "code_source.root_path '$ROOT_PATH' not found" >&2; exit 1; }

hr() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

hr "1. Scratch copy: train.yaml + $ROOT_PATH -> $OUT"
rm -rf "$OUT" && mkdir -p "$OUT/$(dirname "$ROOT_PATH")"
cp train.yaml "$OUT/"
cp -R "$ROOT_PATH" "$OUT/$ROOT_PATH"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +

hr "2. databricks air convert-to-dabs train.yaml --output-dir $OUT"
(cd "$OUT" && databricks air convert-to-dabs train.yaml --output-dir . -p "$PROFILE")

hr "3. Generated $OUT/databricks.yml"
cat "$OUT/databricks.yml"
hr "   Generated generated_artifacts/command.sh (= train.yaml 'command')"
cat "$OUT/generated_artifacts/command.sh"
hr "   Generated generated_artifacts/env_vars.json (= train.yaml env_variables; NOT referenced by the bundle)"
cat "$OUT/generated_artifacts/env_vars.json"; echo

hr "4. diff: generated task (left) vs committed 'train' task in $JOB_FILE (right)"
if command -v yq >/dev/null 2>&1; then
  yq -P 'sort_keys(..) | .resources.jobs[].tasks[0] | del(.task_key, .environment_key)' \
    "$OUT/databricks.yml" > "$OUT/generated_task.yml"
  yq -P 'sort_keys(..) | .resources.jobs[].tasks[] | select(.task_key == "train")
         | del(.task_key, .environment_key, .depends_on) | ... comments=""' \
    "$JOB_FILE" > "$OUT/committed_task.yml"
  yq -P 'sort_keys(..) | .resources.jobs[].environments | map(del(.environment_key))' "$OUT/databricks.yml" > "$OUT/generated_env.yml"
  yq -P 'sort_keys(..) | .resources.jobs[].environments | map(del(.environment_key)) | ... comments=""' "$JOB_FILE" > "$OUT/committed_env.yml"
  diff -u --label generated --label committed "$OUT/generated_task.yml" "$OUT/committed_task.yml" || true
  hr "   environments"
  diff -u --label generated --label committed "$OUT/generated_env.yml" "$OUT/committed_env.yml" || true
  cat <<'EOF'

Expected differences (deliberate, documented in the job YAML):
  * code_source_path: ./dist/train_code.tgz   same train/ layout + run_config.env (bundle vars)
  * command_path: ../src/train/command.sh     the generated wrapper only cd's into the code and runs it
  * experiment: ${var.experiment}             same default (mlb-air-yolo), overridable per target
  * task_key / environment_key / depends_on   local naming and DAG position (not diffed)
EOF
else
  echo "(yq not installed: showing the committed job file instead of a normalized diff)"
  cat "$JOB_FILE"
fi

if [ "$VALIDATE" = 1 ]; then
  hr "5. databricks bundle validate (generated scratch bundle, read-only; nothing is deployed)"
  (cd "$OUT" && databricks bundle validate -p "$PROFILE")
fi

hr "Done. Nothing was deployed; repo files are unchanged. Scratch output: $OUT"
