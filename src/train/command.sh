#!/bin/bash
# Launch script for the DABs ai_runtime_task. It runs once per GPU node.
# The code archive (src/train) is unpacked on the node, under $CODE_SOURCE_PATH/train.
set -euo pipefail
for d in "${CODE_SOURCE_PATH:-}/train" "${CODE_SOURCE_PATH:-}" "$PWD" "$PWD/train"; do
  if [ -n "$d" ] && [ -f "$d/train.py" ]; then cd "$d"; break; fi
done
echo "code dir: $PWD"
python -c "import torch, ultralytics, mlflow; print('torch', torch.__version__, '| ultralytics', ultralytics.__version__, '| mlflow', mlflow.__version__, '| cuda', torch.cuda.is_available())"
python -u train.py --epochs 30 --base_model yolov8s.pt
