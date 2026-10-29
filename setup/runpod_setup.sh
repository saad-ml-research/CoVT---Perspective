#!/usr/bin/env bash
# One-time setup on a RunPod pod. Everything heavy lives on the /workspace volume
# so it survives pod restarts.
#
#   cd /workspace/CoVT---Perspective && bash setup/runpod_setup.sh
set -euo pipefail

WS=${WS:-/workspace}
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
THIRD=$REPO_DIR/third_party
UPSTREAM_COMMIT=c3497a325b0fdda4d91ade456a01b0e325ee3150   # Wakals/CoVT, pinned

export HF_HOME=$WS/hf_cache
export PIP_CACHE_DIR=$WS/pip_cache
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$THIRD"

# Persist env vars for new shells.
grep -q "HF_HOME=$WS/hf_cache" ~/.bashrc 2>/dev/null || {
  echo "export HF_HOME=$WS/hf_cache" >> ~/.bashrc
  echo "export PIP_CACHE_DIR=$WS/pip_cache" >> ~/.bashrc
}

echo ">>> Python: $(which python) ($(python --version))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 1. Upstream CoVT (its VLMEvalKit fork knows the CoVT models).
if [ ! -d "$THIRD/CoVT/.git" ]; then
  git clone https://github.com/Wakals/CoVT.git "$THIRD/CoVT"
fi
git -C "$THIRD/CoVT" fetch --depth 1 origin "$UPSTREAM_COMMIT" || true
git -C "$THIRD/CoVT" checkout -q "$UPSTREAM_COMMIT" || echo "!! could not pin upstream commit; using current HEAD"

python "$REPO_DIR/setup/patch_upstream.py"   # sdpa fallback when flash-attn is absent

# 2. Python deps. Versions match CoVT's training requirements.
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.50.1 accelerate==1.3.0 qwen-vl-utils pandas pyarrow \
            datasets scikit-learn tqdm pillow "huggingface_hub[cli]"
pip install -e "$THIRD/CoVT/VLMEvalKit"
# flash-attn is optional (code falls back to sdpa); it can take a while to build.
pip install flash-attn==2.7.4.post1 --no-build-isolation || echo "!! flash-attn failed; will use sdpa"

# 3. This repo as a package.
pip install -e "$REPO_DIR"

# 4. Re-pin last: later installs can pull a newer transformers, which removes
#    classes VLMEvalKit imports (e.g. AutoModelForVision2Seq).
pip install "transformers==4.50.1" "accelerate==1.3.0"
python -c "import transformers as t; assert t.__version__=='4.50.1', t.__version__; from transformers import AutoModelForVision2Seq; print('transformers', t.__version__, 'ok')"

python "$REPO_DIR/scripts/check_env.py"
echo ">>> Setup done. Next: bash setup/download_models.sh"
