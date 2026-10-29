"""Convert VERIFY to the project JSONL schema (data/SCHEMA.md).

Public release so far: jing-bi/verify-teaser (50 puzzles, test split) with
columns uid, image, question, options (4), reasoning, answer, year. The full
600-puzzle set is not public yet; point --repo at it once it is.

    python scripts/prepare_verify.py --out data/verify_teaser.jsonl
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path


def answer_to_letter(answer: str, options: list[str]) -> str | None:
    """VERIFY's answer field may be a letter or the option text; normalize to a letter."""
    letters = string.ascii_uppercase[: len(options)]
    a = str(answer).strip()
    if a.upper() in letters and len(a) == 1:
        return a.upper()
    a_clean = a.strip("() .").upper()
    if a_clean in letters and len(a_clean) == 1:
        return a_clean
    for letter, opt in zip(letters, options):
        if str(opt).strip().lower() == a.lower():
            return letter
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="jing-bi/verify-teaser")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("data/verify_teaser.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.repo, split=args.split)
    img_dir = args.out.parent / "images" / "verify"
    img_dir.mkdir(parents=True, exist_ok=True)

    n_bad = 0
    with args.out.open("w") as f:
        for row in ds:
            options = list(row["options"])
            letters = string.ascii_uppercase[: len(options)]
            letter = answer_to_letter(row["answer"], options)
            if letter is None:
                n_bad += 1
                print(f"!! could not map answer {row['answer']!r} for {row['uid']}")
            img_path = img_dir / f"{row['uid']}.png"
            if not img_path.exists():
                row["image"].convert("RGB").save(img_path)
            rec = {
                "id": f"verify_{row['uid']}",
                "task": row.get("category") or "VERIFY",   # pattern type not in the teaser
                "images": [str(img_path.relative_to(args.out.parent))],
                "question": row["question"],
                "options": dict(zip(letters, options)),
                "answer": letter,
                "meta": {"reasoning": row.get("reasoning"), "year": row.get("year"),
                         "raw_answer": row["answer"]},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(ds)} rows to {args.out} ({n_bad} unmapped answers)")


if __name__ == "__main__":
    main()
