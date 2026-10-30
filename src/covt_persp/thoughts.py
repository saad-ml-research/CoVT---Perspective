"""How often does a CoVT run actually emit visual-thought tokens, and does accuracy
differ when it does?

    python -m covt_persp.thoughts /workspace/results/mmp/full/CoVT-*.jsonl

Per task: share of answers containing visual tokens, which experts appear, and
accuracy with vs. without visual thoughts. Note: the model chooses when to think,
so the with/without split is correlational, not causal.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PADS = {"sam": "<|sam_pad|>", "depth": "<|depth_pad|>", "dino": "<|dino_pad|>", "edge": "<|edge_pad|>"}


def correct(r: dict) -> int:
    return int(r["pred"] is not None and str(r["pred"]).upper() == str(r["answer"]).strip().upper())


def analyze(path: Path) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r.get("output_raw") or r.get("output") or ""
        r["_experts"] = [k for k, tok in PADS.items() if tok in raw]
        groups[r.get("task") or "all"].append(r)
        groups["ALL"].append(r)
    out = {}
    for task, rs in sorted(groups.items()):
        w = [r for r in rs if r["_experts"]]
        wo = [r for r in rs if not r["_experts"]]
        out[task] = {
            "n": len(rs),
            "think_share": len(w) / len(rs),
            "acc_with": sum(map(correct, w)) / len(w) if w else None,
            "acc_without": sum(map(correct, wo)) / len(wo) if wo else None,
            "n_with": len(w), "n_without": len(wo),
            "experts": {k: sum(k in r["_experts"] for r in rs) / len(rs) for k in PADS},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    fmt = lambda x: "  -- " if x is None else f"{100 * x:5.1f}"
    for p in args.files:
        s = analyze(p)
        used = [k for k in PADS if s["ALL"]["experts"][k] > 0]
        print(f"\n== {p.name}")
        print(f"  {'task':6s} {'n':>5s}  {'thinks':>6s}  {'acc w/':>6s} {'(n)':>5s}  {'acc w/o':>7s} {'(n)':>5s}  "
              + "  ".join(f"{k:>5s}" for k in used))
        for task, v in s.items():
            print(f"  {task:6s} {v['n']:5d}  {100 * v['think_share']:5.1f}%  {fmt(v['acc_with'])} {v['n_with']:5d}"
                  f"  {fmt(v['acc_without']):>7s} {v['n_without']:5d}  "
                  + "  ".join(f"{100 * v['experts'][k]:4.0f}%" for k in used))


if __name__ == "__main__":
    main()
