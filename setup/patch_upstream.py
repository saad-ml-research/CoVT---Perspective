"""Make upstream VLMEvalKit wrappers fall back to SDPA attention when flash-attn
is not installed. Idempotent; run after cloning third_party/CoVT.

    python setup/patch_upstream.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "CoVT" / "VLMEvalKit" / "vlmeval" / "vlm"
FILES = [ROOT / "covt_qwen" / "model.py", ROOT / "qwen2_vl" / "model.py"]
OLD = "attn_implementation='flash_attention_2'"
NEW = ("attn_implementation=('flash_attention_2' if __import__('importlib.util').util"
       ".find_spec('flash_attn') else 'sdpa')")

for f in FILES:
    s = f.read_text()
    n = s.count(OLD)
    if n:
        f.write_text(s.replace(OLD, NEW))
    print(f"{f.relative_to(ROOT.parents[3])}: patched {n}, already patched {s.count(NEW)}")
