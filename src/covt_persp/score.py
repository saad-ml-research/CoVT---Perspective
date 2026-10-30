"""Answer extraction and accuracy with bootstrap confidence intervals.

    python -m covt_persp.score /workspace/results/mmp/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
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
    """Per task: accuracy with CI, majority-class baseline (always answering the most
    common correct letter), and the model's top-prediction share (1.0 = it always
    gives the same letter, i.e. collapsed onto one option)."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("task") or "all"].append(r)
        if r.get("task"):
            groups["ALL"].append(r)
    out = {}
    for task, rs in sorted(groups.items()):
        c = [int(r["pred"] is not None and str(r["pred"]).upper() == str(r["answer"]).strip().upper())
             for r in rs]
        lo, hi = bootstrap_ci(c)
        answers = Counter(str(r["answer"]).upper() for r in rs)
        preds = Counter(str(r["pred"]).upper() for r in rs)
        top_pred, top_n = preds.most_common(1)[0]
        out[task] = {
            "n": len(c), "acc": sum(c) / len(c), "ci95": (lo, hi),
            "majority": answers.most_common(1)[0][1] / len(rs),
            "top_pred": top_pred, "top_pred_share": top_n / len(rs),
        }
    out["_unparsed"] = sum(r["pred"] is None for r in rows)
    return out


def agreement(a: Path, b: Path) -> float:
    """Fraction of shared questions where two runs give the same letter."""
    pa = {r["id"]: r["pred"] for r in map(json.loads, a.read_text().splitlines()) if r}
    pb = {r["id"]: r["pred"] for r in map(json.loads, b.read_text().splitlines()) if r}
    shared = pa.keys() & pb.keys()
    return sum(pa[i] == pb[i] for i in shared) / max(len(shared), 1)


def _correct_by_id(path: Path) -> dict[str, tuple[str, int]]:
    out = {}
    for l in path.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            ok = r["pred"] is not None and str(r["pred"]).upper() == str(r["answer"]).strip().upper()
            out[r["id"]] = (r.get("task") or "all", int(ok))
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial sign test on discordant pairs)."""
    from math import comb
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a: Path, b: Path, n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Per task: accuracy difference (b - a), paired bootstrap 95% CI, McNemar p."""
    ca, cb = _correct_by_id(a), _correct_by_id(b)
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i in ca.keys() & cb.keys():
        pair = (ca[i][1], cb[i][1])
        groups[ca[i][0]].append(pair)
        groups["ALL"].append(pair)
    rng = random.Random(seed)
    out = {}
    for task, ps in sorted(groups.items()):
        n = len(ps)
        diff = sum(y - x for x, y in ps) / n
        boots = sorted(sum(y - x for x, y in rng.choices(ps, k=n)) / n for _ in range(n_boot))
        only_a = sum(x and not y for x, y in ps)
        only_b = sum(y and not x for x, y in ps)
        out[task] = {"n": n, "diff": diff, "ci95": (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]),
                     "only_a": only_a, "only_b": only_b, "p": mcnemar_p(only_a, only_b)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    for p in args.files:
        s = summarize(p)
        print(f"\n== {p.name}  (unparsed: {s.pop('_unparsed')})")
        print(f"  {'task':6s} {'n':>5s}  {'acc':>5s}  {'95% CI':>13s}  {'majority':>8s}  {'top pred':>9s}")
        for task, v in s.items():
            lo, hi = v["ci95"]
            flag = "  <- collapsed" if v["top_pred_share"] >= 0.8 and v["n"] >= 5 else ""
            print(f"  {task:6s} {v['n']:5d}  {100 * v['acc']:5.1f}  [{100 * lo:4.1f}, {100 * hi:4.1f}]"
                  f"  {100 * v['majority']:8.1f}  {v['top_pred']:>3s} {100 * v['top_pred_share']:4.0f}%{flag}")
    if len(args.files) == 2:
        a, b = args.files
        print(f"\nprediction agreement between the two runs: {100 * agreement(a, b):.1f}%")
        print(f"\n== paired: {b.stem} minus {a.stem}")
        print(f"  {'task':6s} {'n':>5s}  {'diff':>6s}  {'95% CI':>15s}  {'only A':>6s}  {'only B':>6s}  {'McNemar p':>9s}")
        for task, v in paired(a, b).items():
            lo, hi = v["ci95"]
            star = "  *" if v["p"] < 0.05 else ""
            print(f"  {task:6s} {v['n']:5d}  {100 * v['diff']:+6.1f}  [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
                  f"  {v['only_a']:6d}  {v['only_b']:6d}  {v['p']:9.4f}{star}")


if __name__ == "__main__":
    main()
