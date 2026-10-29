"""Detailed CV-Bench breakdown from VLMEvalKit's per-question prediction files.

VLMEvalKit's *_acc.csv only splits 2D by image source and 3D by task. This reads the
per-question files and reports accuracy by task, by source, and by task x source,
with bootstrap 95% CIs, plus the CV-Bench overall (mean of 2D and 3D).

    python scripts/cvbench_breakdown.py --work-dir /workspace/results/vlmevalkit \
        --models CoVT-7B-seg_depth_dino Qwen2.5-VL-7B-Instruct --markdown
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from covt_persp.score import bootstrap_ci, extract_letter

TASK_COLS = ["task", "category", "type"]
SOURCE_COLS = ["source", "split", "dataset"]


def pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[c] for c in cands if c in lower), None)


def load_scored(model_dir: Path, model: str, dataset: str) -> pd.DataFrame:
    """Prefer a file with VLMEvalKit's 'hit' column; else score raw predictions."""
    files = sorted(model_dir.glob(f"{model}_{dataset}*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not files:
        files = sorted(model_dir.rglob(f"{model}_{dataset}*.xlsx"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no prediction xlsx for {model} x {dataset} under {model_dir}")
    for f in reversed(files):
        df = pd.read_excel(f)
        if "hit" in df.columns:
            df["correct"] = df["hit"].astype(int)
            df.attrs["file"] = f
            return df
    df = pd.read_excel(files[-1])
    letters = [c for c in "ABCDEF" if c in df.columns]
    df["correct"] = [
        int(extract_letter(str(p), letters) == str(a).strip().upper())
        for p, a in zip(df["prediction"], df["answer"])
    ]
    df.attrs["file"] = files[-1]
    return df


def acc_rows(df: pd.DataFrame, by: list[str]) -> list[tuple]:
    out = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        c = g["correct"].tolist()
        lo, hi = bootstrap_ci(c)
        out.append((" / ".join(map(str, key)), len(c), 100 * sum(c) / len(c), 100 * lo, 100 * hi))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path("/workspace/results/vlmevalkit"))
    ap.add_argument("--models", nargs="+", default=["CoVT-7B-seg_depth_dino"])
    ap.add_argument("--markdown", action="store_true", help="print markdown tables for docs/results.md")
    args = ap.parse_args()

    for model in args.models:
        overall = {}
        print(f"\n### {model}")
        for ds in ["CV-Bench-2D", "CV-Bench-3D"]:
            try:
                df = load_scored(args.work_dir / model, model, ds)
            except FileNotFoundError as e:
                print(f"  ({e})")
                continue
            t, s = pick(df, TASK_COLS), pick(df, SOURCE_COLS)
            if not (t and s):
                print(f"  !! task col={t}, source col={s}; columns are: {list(df.columns)}")
            c = df["correct"].tolist()
            overall[ds] = 100 * sum(c) / len(c)
            print(f"\n{ds}  (n={len(c)}, acc={overall[ds]:.1f})  file: {df.attrs['file'].name}")
            groups = [[g] for g in (t, s) if g] + ([[t, s]] if t and s else [])
            if args.markdown:
                print("\n| Split | n | Acc | 95% CI |\n|---|---|---|---|")
            for by in groups:
                for name, n, a, lo, hi in acc_rows(df, by):
                    if args.markdown:
                        print(f"| {name} | {n} | {a:.1f} | [{lo:.1f}, {hi:.1f}] |")
                    else:
                        print(f"  {name:28s} n={n:5d}  acc={a:5.1f}  [{lo:5.1f}, {hi:5.1f}]")
        if len(overall) == 2:
            print(f"\nCV-Bench overall (mean of 2D, 3D): {(overall['CV-Bench-2D'] + overall['CV-Bench-3D']) / 2:.1f}")


if __name__ == "__main__":
    main()
