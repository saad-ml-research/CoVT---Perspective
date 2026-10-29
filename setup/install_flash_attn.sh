#!/usr/bin/env bash
# Install a prebuilt flash-attn 2.7.4.post1 wheel matching this pod's torch / Python /
# C++ ABI, instead of compiling from source (slow and often fails).
# Needed for memory: without it the Qwen2.5-VL vision encoder's SDPA attention OOMs
# on large images at CoVT's max_pixels setting.
set -euo pipefail

read -r TORCH_MM CUDA_MAJ PY ABI < <(python - <<'EOF'
import sys, torch
mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cuda = (torch.version.cuda or "12").split(".")[0]
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(mm, cuda, py, abi)
EOF
)
VER=2.7.4.post1
WHEEL="flash_attn-${VER}+cu${CUDA_MAJ}torch${TORCH_MM}cxx11abi${ABI}-${PY}-${PY}-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${VER}/${WHEEL}"

echo ">>> torch ${TORCH_MM}  cuda ${CUDA_MAJ}  ${PY}  cxx11abi=${ABI}"
echo ">>> ${URL}"
pip install --no-deps "${URL}"
python -c "import flash_attn, torch; print('flash_attn', flash_attn.__version__, 'ok on', torch.cuda.get_device_name(0))"
