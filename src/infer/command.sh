#!/bin/bash
# Launch script for the GPU batch-inference ai_runtime_task. The code archive
# (src/infer) is unpacked on the node, under $CODE_SOURCE_PATH/infer.
set -euo pipefail
for d in "${CODE_SOURCE_PATH:-}/infer" "${CODE_SOURCE_PATH:-}" "$PWD" "$PWD/infer"; do
  if [ -n "$d" ] && [ -f "$d/infer.py" ]; then cd "$d"; break; fi
done
echo "code dir: $PWD"
python -u infer.py --max_images 60
