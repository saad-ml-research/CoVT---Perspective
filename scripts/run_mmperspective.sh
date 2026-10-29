#!/usr/bin/env bash
# Part A on MMPerspective: every model x condition. Resumable; re-run to continue.
#   bash scripts/run_mmperspective.sh smoke     # 5 questions per task, quick check
#   bash scripts/run_mmperspective.sh           # full run (perception + reasoning tasks)
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$REPO_DIR/data/mmperspective.jsonl}
MODE=${1:-full}
OUT=${OUT:-/workspace/results/mmp/$MODE}
MODELS=${MODELS:-"Qwen/Qwen2.5-VL-7B-Instruct Wakals/CoVT-7B-seg_depth_dino"}
CONDITIONS=${CONDITIONS:-"direct"}

EXTRA=(--exclude-tasks PRB)              # robustness scored separately (consistency)
[ "$MODE" = smoke ] && EXTRA+=(--per-task 5)

for m in $MODELS; do
  for c in $CONDITIONS; do
    tag="$(basename "$m")__$c"
    echo ">>> $tag"
    python -m covt_persp.infer --model "$m" --data "$DATA" --condition "$c" \
      --out "$OUT/$tag.jsonl" "${EXTRA[@]}"
  done
done
python -m covt_persp.score "$OUT"/*.jsonl
