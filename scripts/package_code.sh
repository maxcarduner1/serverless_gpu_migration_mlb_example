#!/bin/bash
# Package src/<component> as dist/<component>_code.tgz for an ai_runtime_task.
# Runs as the bundle artifact build step:
#   bash scripts/package_code.sh <component> <catalog> <schema> <model_name>
# It writes <component>/run_config.env (CATALOG, SCHEMA, MODEL_NAME) into the archive,
# because ai_runtime_task has no task parameters.
set -euo pipefail
cd "$(dirname "$0")/.."
component="$1" catalog="$2" schema="$3" model_name="$4"
for v in "$component" "$catalog" "$schema" "$model_name"; do
  [[ "$v" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid identifier: '$v'" >&2; exit 1; }
done
stage="dist/stage_${component}"
rm -rf "dist/stage_${component}"
mkdir -p "$stage"
cp -R "src/${component}" "$stage/"
find "$stage" -name __pycache__ -prune -exec rm -rf {} +
printf 'CATALOG=%s\nSCHEMA=%s\nMODEL_NAME=%s\n' "$catalog" "$schema" "$model_name" > "$stage/${component}/run_config.env"
COPYFILE_DISABLE=1 tar -czf "dist/${component}_code.tgz" -C "$stage" "$component"
rm -rf "dist/stage_${component}"
echo "packaged dist/${component}_code.tgz (catalog=$catalog schema=$schema model=$model_name)"
