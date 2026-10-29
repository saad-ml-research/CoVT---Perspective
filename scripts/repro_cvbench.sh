#!/usr/bin/env bash
# Step 0: confirm the pipeline reproduces CoVT's CV-Bench number (paper: 80.0
# for CoVT-7B-seg_depth_dino, 74.5 for the Qwen2.5-VL-7B baseline).
#   bash scripts/repro_cvbench.sh                       # CoVT main model
#   bash scripts/repro_cvbench.sh Qwen2.5-VL-7B-Instruct
set -euo pipefail
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-CoVT-7B-seg_depth_dino}
OUT=${OUT:-/workspace/results/vlmevalkit}
mkdir -p "$OUT"

# The CoVT wrapper resolves paths relative to the VLMEvalKit dir, so run from there.
cd "$REPO_DIR/third_party/CoVT/VLMEvalKit"
python run.py --data CV-Bench-2D CV-Bench-3D --model "$MODEL" --work-dir "$OUT" --verbose

echo ">>> Results:"
find "$OUT/$MODEL" -name "*acc*.csv" -exec sh -c 'echo "== $1"; cat "$1"' _ {} \;
