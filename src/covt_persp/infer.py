"""Run a Qwen2.5-VL-family model (base or CoVT checkpoint) over a JSONL benchmark.

Input: one JSON object per line (see data/SCHEMA.md):
    {"id", "task", "images": [paths], "question", "options": {"A": ...}, "answer"}

Output: JSONL with the raw generation, the decoded text with special tokens kept
(so CoVT's visual-token placeholders stay visible), and the extracted letter.
Resumable: rows already in the output file are skipped.

    python -m covt_persp.infer --model Wakals/CoVT-7B-seg_depth_dino \
        --data data/mmperspective.jsonl --condition text_cot \
        --out /workspace/results/mmp/covt_sdd__text_cot.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .prompts import CONDITIONS, build_prompt
from .score import extract_letter

# CoVT's VLMEvalKit config uses these pixel bounds for every model. Override with
# COVT_MAX_PIXELS only if you must (e.g. no flash-attn); note it in results if so.
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = int(os.environ.get("COVT_MAX_PIXELS", 16384 * 28 * 28))


def load_model(model_path: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"loaded {model_path} (attn={attn})")
    return model, processor


@torch.inference_mode()
def generate(model, processor, images: list[str], prompt: str, max_new_tokens: int) -> tuple[str, str]:
    from qwen_vl_utils import process_vision_info

    content = [{"type": "image", "image": str(p), "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
               for p in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, inputs.input_ids.shape[1]:]
    clean = processor.decode(new, skip_special_tokens=True)
    raw = processor.decode(new, skip_special_tokens=False)
    return clean, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only")
    ap.add_argument("--per-task", type=int, default=None, help="first N rows of each task (smoke test)")
    ap.add_argument("--tasks", nargs="+", default=None, help="only these task codes")
    ap.add_argument("--exclude-tasks", nargs="+", default=None, help="skip these task codes")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    # CoVT writes ~50 tokens of visual thoughts before <answer>, even in direct mode.
    max_new = args.max_new_tokens or (256 if args.condition == "direct" else 1024)
    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    if args.tasks:
        rows = [r for r in rows if r.get("task") in set(args.tasks)]
    if args.exclude_tasks:
        rows = [r for r in rows if r.get("task") not in set(args.exclude_tasks)]
    if args.per_task:
        seen: dict[str, int] = {}
        keep = []
        for r in rows:
            t = r.get("task")
            if seen.get(t, 0) < args.per_task:
                keep.append(r)
                seen[t] = seen.get(t, 0) + 1
        rows = keep
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to run")
    if not todo:
        return

    model, processor = load_model(args.model)
    data_root = args.data.parent
    with args.out.open("a") as f:
        for r in tqdm(todo):
            images = [p if Path(p).is_absolute() else str(data_root / p) for p in r["images"]]
            prompt = build_prompt(r["question"], r.get("options"), args.condition)
            t0 = time.time()
            clean, raw = generate(model, processor, images, prompt, max_new)
            rec = {
                "id": r["id"], "task": r.get("task"), "answer": r.get("answer"),
                "pred": extract_letter(clean, list((r.get("options") or {}).keys())),
                "output": clean, "output_raw": raw,
                "model": args.model, "condition": args.condition,
                "n_new_tokens": len(processor.tokenizer(raw).input_ids),
                "truncated": len(processor.tokenizer(raw).input_ids) >= max_new,
                "seconds": round(time.time() - t0, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
