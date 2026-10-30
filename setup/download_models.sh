#!/usr/bin/env bash
# Download checkpoints to the /workspace volume (~16 GB each).
#   bash setup/download_models.sh            # main CoVT model + base
#   bash setup/download_models.sh all        # all five Qwen-based models (~80 GB)
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

MAIN=(Wakals/CoVT-7B-seg_depth_dino Qwen/Qwen2.5-VL-7B-Instruct)
REST=(Wakals/CoVT-7B-seg Wakals/CoVT-7B-depth Wakals/CoVT-7B-seg_depth_dino_edge)

MODELS=("${MAIN[@]}")
[ "${1:-}" = "all" ] && MODELS+=("${REST[@]}")

df -h /workspace | tail -1
for m in "${MODELS[@]}"; do
  echo ">>> $m"
  huggingface-cli download "$m" --exclude "*.pth" "*.pt"
done
du -sh "$HF_HOME"
