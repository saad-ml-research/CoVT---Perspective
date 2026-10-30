#!/usr/bin/env bash
# Part A on MMPerspective: every model x condition. Resumable; re-run to continue.
#   bash scripts/run_mmperspective.sh smoke     # 5 questions per task, quick check
#   bash scripts/run_mmperspective.sh           # full run (perception + reasoning tasks)
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$REPO_DIR/data/mmperspective.jsonl}
MODE=${1:-full}
DNAME=$(basename "${DATA%.jsonl}"); [ "$DNAME" = mmperspective ] && DNAME=mmp     # keep existing results path
OUT=${OUT:-/workspace/results/$DNAME/$MODE}
MODELS=${MODELS:-"Qwen/Qwen2.5-VL-7B-Instruct Wakals/CoVT-7B-seg_depth_dino"}
CONDITIONS=${CONDITIONS:-"direct"}
THOUGHTS=${THOUGHTS:-auto}               # CoVT: auto | off | on
TASKS=${TASKS:-}                         # e.g. TASKS="PTR LDP"; empty = all
SUFFIX=${SUFFIX:-}                       # appended to output names, e.g. _minpixdefault

EXTRA=(--thoughts "$THOUGHTS"); [ "${DATA##*/}" = mmperspective.jsonl ] && EXTRA+=(--exclude-tasks PRB)   # robustness scored separately (consistency)
[ -n "$TASKS" ] && EXTRA+=(--tasks $TASKS)
[ "$MODE" = smoke ] && EXTRA+=(--per-task 5)

for m in $MODELS; do
  for c in $CONDITIONS; do
    tag="$(basename "$m")__$c"
    [ "$THOUGHTS" != auto ] && tag="${tag}__think-$THOUGHTS"
    tag="$tag$SUFFIX"
    echo ">>> $tag"
    python -m covt_persp.infer --model "$m" --data "$DATA" --condition "$c" \
      --out "$OUT/$tag.jsonl" "${EXTRA[@]}"
  done
done
echo ">>> done; score with: python -m covt_persp.score <base.jsonl> <run.jsonl>"
