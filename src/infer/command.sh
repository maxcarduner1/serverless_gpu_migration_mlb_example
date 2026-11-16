#!/bin/bash
# GPU batch-inference launcher for the ai_runtime_task. Catalog, schema, and model come
# from run_config.env (written into the code archive from the bundle variables) or from
# env vars CATALOG / SCHEMA / MODEL_NAME. There are no defaults here.
set -euo pipefail
for d in "${CODE_SOURCE_PATH:-}/infer" "${CODE_SOURCE_PATH:-}" "$PWD" "$PWD/infer"; do
  if [ -n "$d" ] && [ -f "$d/infer.py" ]; then cd "$d"; break; fi
done
if [ -f run_config.env ]; then set -a; source run_config.env; set +a; fi
: "${CATALOG:?CATALOG not set}" "${SCHEMA:?SCHEMA not set}" "${MODEL_NAME:?MODEL_NAME not set}"
echo "code dir: $PWD | catalog=$CATALOG schema=$SCHEMA model=$MODEL_NAME"
python -u infer.py --catalog "$CATALOG" --schema "$SCHEMA" --model_name "$MODEL_NAME" --max_images 60
