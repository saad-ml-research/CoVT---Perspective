"""Print the versions and GPU facts that matter for this project."""
import importlib
import os
import shutil

import torch

print(f"torch {torch.__version__}  cuda {torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")

for mod in ["transformers", "accelerate", "qwen_vl_utils", "flash_attn", "vlmeval"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:14s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"{mod:14s} MISSING ({type(e).__name__})")

total, used, free = shutil.disk_usage("/workspace" if os.path.exists("/workspace") else "/")
print(f"disk free {free / 2**30:.0f} GiB of {total / 2**30:.0f} GiB")
