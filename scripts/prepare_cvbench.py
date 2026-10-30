"""Convert CV-Bench (nyu-visionx/CV-Bench) to the project JSONL schema, so it can be
run with covt_persp.infer (needed for the visual-thought on/off control on CV-Bench).

    python scripts/prepare_cvbench.py --out data/cvbench.jsonl

Task codes: Count, Relation (2D) and Depth, Distance (3D). meta keeps type/source.
"""
from __future__ import annotations

import argparse
import json
import string
from collections import Counter
from pathlib import Path

TASK_CODES = {"count": "Count", "relation": "Relation", "depth": "Depth", "distance": "Distance"}


def main() -> None:
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nyu-visionx/CV-Bench")
    ap.add_argument("--out", type=Path, default=Path("data/cvbench.jsonl"))
    args = ap.parse_args()

    img_dir = args.out.parent / "images" / "cvbench"
    img_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.repo, split="test")
    print("columns:", ds.column_names)

    rows, problems = [], Counter()
    for i, r in enumerate(ds):
        rid = str(r.get("idx", i))
        img_path = img_dir / f"{rid}.png"
        if not img_path.exists():
            r["image"].convert("RGB").save(img_path)
        choices = list(r["choices"])
        options = {string.ascii_uppercase[k]: str(c) for k, c in enumerate(choices)}
        ans = str(r["answer"]).strip("() .").upper()
        if ans not in options:
            problems["unmapped_answer"] += 1
            ans = None
        task = TASK_CODES.get(str(r.get("task", "")).lower(), str(r.get("task")))
        rows.append({
            "id": f"cvb_{rid}",
            "task": task,
            "images": [str(img_path.relative_to(args.out.parent))],
            "question": r["question"],
            "options": options,
            "answer": ans,
            "meta": {"type": r.get("type"), "source": r.get("source_dataset") or r.get("source"),
                     "prompt": r.get("prompt")},
        })
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    print("per task:", dict(Counter(r["task"] for r in rows)))
    print("problems:", dict(problems) or "none")


if __name__ == "__main__":
    main()
