#!/bin/bash
# Launch script for the DABs ai_runtime_task. It runs once per GPU node.
# The code archive (src/train) is unpacked on the node. Find train.py,
# whether the launcher starts us in the archive root or exposes it
# through $CODE_SOURCE.
set -euo pipefail
for d in "${CODE_SOURCE:-}" "${CODE_SOURCE:-}/train" "${CODE_SOURCE_PATH:-}" "$PWD" "$PWD/train"; do
  if [ -n "$d" ] && [ -f "$d/train.py" ]; then cd "$d"; break; fi
done
echo "code dir: $PWD"; ls -la
python train.py --epochs 30 --base_model yolov8s.pt
