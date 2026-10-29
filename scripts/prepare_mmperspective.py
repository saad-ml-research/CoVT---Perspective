"""Download MMPerspective (yoloytang/MMPerspective) and convert it to the project
JSONL schema (data/SCHEMA.md).

The Hub viewer shows images in root_v3/<task>/ folders; the QA annotations appear
to be JSON files alongside them. Step 1 inspects the layout, step 2 converts.

    python scripts/prepare_mmperspective.py --inspect        # paste this output back
    python scripts/prepare_mmperspective.py --out data/mmperspective.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import string
from collections import Counter
from pathlib import Path

TASK_CODES = {
    "vanishing_point_perception": "VPP",
    "critical_line_perception": "CLP",
    "view_angle_perception": "VAP",
    "lens_distortion_perception": "LDP",
    "perspective_type_reasoning": "PTR",
    "line_relationship_reasoning": "LRR",
    "perspective_transformation_spotting": "PTS",
    "vanishing_point_counting": "VPC",
    "out_of_view_reasoning": "OVR",
    "perspective_robustness": "PRB",
}
CATEGORY = {"VPP": "perception", "CLP": "perception", "VAP": "perception", "LDP": "perception",
            "PTR": "reasoning", "LRR": "reasoning", "PTS": "reasoning", "VPC": "reasoning",
            "OVR": "reasoning", "PRB": "robustness"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def download(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo, repo_type="dataset"))


def load_json_any(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    return json.loads(text)


def inspect(root: Path) -> None:
    print(f"snapshot: {root}")
    for dirpath, dirs, files in sorted(os.walk(root)):
        if "/.git" in dirpath or "/.cache" in dirpath:
            continue
        rel = Path(dirpath).relative_to(root)
        exts = Counter(Path(f).suffix.lower() for f in files)
        print(f"{str(rel) or '.':60s} {dict(exts)}")
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in {".json", ".jsonl"}:
            data = load_json_any(p)
            items = data if isinstance(data, list) else data.get("data", data) if isinstance(data, dict) else data
            sample = items[0] if isinstance(items, list) and items else items
            n = len(items) if isinstance(items, (list, dict)) else "?"
            print(f"\n--- {p.relative_to(root)}  ({type(data).__name__}, n={n})")
            print(json.dumps(sample, ensure_ascii=False, indent=1)[:1500])


def records_from(path: Path) -> list[dict]:
    data = load_json_any(path)
    if isinstance(data, dict):
        for k in ("data", "items", "questions", "annotations"):
            if isinstance(data.get(k), list):
                return data[k]
        if all(isinstance(v, dict) for v in data.values()):
            return [{"_key": k, **v} for k, v in data.items()]
        return [data]
    return data


def first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def normalize_options(raw) -> dict[str, str] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k).strip(" .").upper(): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out = {}
        for i, opt in enumerate(raw):
            s = str(opt).strip()
            if len(s) > 2 and s[0].upper() in string.ascii_uppercase and s[1] in ".):":
                out[s[0].upper()] = s[2:].strip()
            else:
                out[string.ascii_uppercase[i]] = s
        return out
    return None


def normalize_answer(ans, options: dict[str, str] | None) -> str | None:
    if ans is None:
        return None
    a = str(ans).strip()
    if options and a.strip("() .").upper() in options:
        return a.strip("() .").upper()
    if options:
        for k, v in options.items():
            if v.strip().lower() == a.lower():
                return k
    return None


def convert(root: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, problems = [], Counter()
    for jpath in sorted(root.rglob("*.json*")):
        task_dir = next((p for p in jpath.relative_to(root).parts if p in TASK_CODES), None)
        for i, r in enumerate(records_from(jpath)):
            task = TASK_CODES.get(task_dir) or TASK_CODES.get(str(first(r, "task", "category"))) \
                or str(first(r, "task", "category") or "unknown")
            imgs = first(r, "images", "image_paths", "image", "image_path", "img", "file_name")
            imgs = imgs if isinstance(imgs, list) else [imgs] if imgs else []
            resolved = []
            for im in imgs:
                cands = [jpath.parent / im, root / im, root / "root_v3" / im]
                hit = next((c for c in cands if c.exists()), None)
                if hit is None:
                    problems["missing_image"] += 1
                resolved.append(str(hit or im))
            options = normalize_options(first(r, "options", "choices", "candidates"))
            answer = normalize_answer(first(r, "answer", "gt", "label", "correct"), options)
            if answer is None:
                problems["unmapped_answer"] += 1
            rid = first(r, "id", "qid", "question_id", "_key") or f"{jpath.stem}_{i}"
            rows.append({
                "id": f"mmp_{task}_{rid}",
                "task": task,
                "images": resolved,
                "question": first(r, "question", "prompt", "query") or "",
                "options": options,
                "answer": answer,
                "meta": {"category": CATEGORY.get(task, "robustness?"), "source_file": str(jpath.relative_to(root)),
                         **{k: v for k, v in r.items() if k not in
                            {"question", "options", "choices", "answer", "images", "image"}}},
            })
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    print("per task:", dict(Counter(r["task"] for r in rows)))
    print("problems:", dict(problems) or "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="yoloytang/MMPerspective")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/mmperspective.jsonl"))
    args = ap.parse_args()
    root = download(args.repo)
    inspect(root) if args.inspect else convert(root, args.out)


if __name__ == "__main__":
    main()
