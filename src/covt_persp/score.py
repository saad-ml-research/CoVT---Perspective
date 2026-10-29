"""Answer extraction and accuracy with bootstrap confidence intervals.

    python -m covt_persp.score /workspace/results/mmp/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ANSWER_TAG = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.I | re.S)
BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def extract_letter(text: str, letters: list[str] | None = None) -> str | None:
    """Pull the chosen option letter out of a model response.

    Order: <answer> tag, then \\boxed{}, then the last standalone option letter.
    """
    letters = letters or list("ABCD")
    pat = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape, letters)) + r")(?![A-Za-z])")

    for source in (ANSWER_TAG.findall(text), BOXED.findall(text)):
        for chunk in reversed(source):
            m = pat.search(chunk.strip())
            if m:
                return m.group(1)
    hits = pat.findall(text)
    return hits[-1] if hits else None


def bootstrap_ci(correct: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not correct:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(correct)
    means = sorted(sum(rng.choices(correct, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summarize(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by_task: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        ok = int(r["pred"] is not None and str(r["pred"]).upper() == str(r["answer"]).strip().upper())
        by_task[r.get("task") or "all"].append(ok)
        if r.get("task"):
            by_task["ALL"].append(ok)
    out = {}
    for task, c in sorted(by_task.items()):
        lo, hi = bootstrap_ci(c)
        out[task] = {"n": len(c), "acc": sum(c) / len(c), "ci95": (lo, hi)}
    out["_unparsed"] = sum(r["pred"] is None for r in rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.files:
        s = summarize(p)
        print(f"\n== {p.name}  (unparsed: {s.pop('_unparsed')})")
        for task, v in s.items():
            lo, hi = v["ci95"]
            print(f"  {task:12s} n={v['n']:5d}  acc={100 * v['acc']:5.1f}  [{100 * lo:5.1f}, {100 * hi:5.1f}]")


if __name__ == "__main__":
    main()
