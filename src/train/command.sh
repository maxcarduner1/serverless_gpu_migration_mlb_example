#!/bin/bash
# GPU launcher, shared by the DABs ai_runtime_task and `databricks air run -f train.yaml`.
# It runs once per GPU node. Catalog, schema, and model come from either:
#   * run_config.env, written into the code archive from the bundle variables, or
#   * env vars CATALOG / SCHEMA / MODEL_NAME (train.yaml `env_variables`).
# There are no defaults here: a missing value fails fast.
set -euo pipefail
for d in "${CODE_SOURCE_PATH:-}/train" "${CODE_SOURCE_PATH:-}" "$PWD" "$PWD/train"; do
  if [ -n "$d" ] && [ -f "$d/train.py" ]; then cd "$d"; break; fi
done
if [ -f run_config.env ]; then set -a; source run_config.env; set +a; fi
: "${CATALOG:?CATALOG not set}" "${SCHEMA:?SCHEMA not set}" "${MODEL_NAME:?MODEL_NAME not set}"
echo "code dir: $PWD | catalog=$CATALOG schema=$SCHEMA model=$MODEL_NAME"
python -c "import torch, ultralytics, mlflow; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| mlflow', mlflow.__version__, '| cuda', torch.cuda.is_available())"
python -u train.py --catalog "$CATALOG" --schema "$SCHEMA" --model_name "$MODEL_NAME" \
  --epochs "${EPOCHS:-30}" --base_model "${BASE_MODEL:-yolov8s.pt}"
