# Visual Thought Tokens: Do They Help MLLMs See, Reason, or Both?

A controlled study of text vs. visual-text reasoning in MLLMs, using the released
[CoVT](https://github.com/Wakals/CoVT) checkpoints on two benchmarks that separate
seeing from reasoning:

- **MMPerspective** (NeurIPS 2025): perspective geometry; perception, reasoning, robustness.
- **VERIFY** (2025): abstract visual puzzles with human reasoning paths, scored by stage.

**Part A** evaluates CoVT as released (visual tokens once, before the answer), zero-shot.
**Part B** trains an interleaved version that emits expert visual tokens before each
text reasoning step.

## Conditions

| Condition | Model | Prompt |
| --- | --- | --- |
| Direct | Qwen2.5-VL-7B-Instruct | `direct` |
| Text reasoning | Qwen2.5-VL-7B-Instruct | `text_cot` |
| Visual thoughts only | CoVT checkpoints | `direct` |
| Visual + text reasoning | CoVT checkpoints | `text_cot` |

## Quick start (RunPod, L40S 48 GB)

```bash
cd /workspace
git clone https://github.com/saad-ml-research/CoVT---Perspective.git
cd CoVT---Perspective
bash setup/runpod_setup.sh          # deps + pinned upstream CoVT into third_party/
bash setup/download_models.sh       # CoVT-7B-seg_depth_dino + Qwen2.5-VL-7B (~32 GB)
bash scripts/repro_cvbench.sh       # Step 0: expect about 80.0 on CV-Bench
bash scripts/repro_cvbench.sh Qwen2.5-VL-7B-Instruct   # baseline, expect about 74.5
```

Then, once benchmarks are converted to `data/*.jsonl` (see `data/SCHEMA.md`):

```bash
python -m covt_persp.infer --model Wakals/CoVT-7B-seg_depth_dino \
    --data data/mmperspective.jsonl --condition text_cot --limit 20 \
    --out /workspace/results/mmp/covt_sdd__text_cot.jsonl
python -m covt_persp.score /workspace/results/mmp/*.jsonl
```

All results and caches live on `/workspace` so they survive pod restarts.

## Layout

```
setup/      RunPod setup and model downloads
scripts/    environment check, CV-Bench reproduction, benchmark converters
src/covt_persp/
  prompts.py   reasoning conditions
  infer.py     resumable runner (keeps raw output incl. visual-token placeholders)
  score.py     answer extraction, per-task accuracy, bootstrap 95% CIs
data/       JSONL schema; converted benchmarks (not committed)
```

## Data status (Oct 2026)

- **VERIFY:** only the 50-puzzle teaser is public (`jing-bi/verify-teaser`).
  `python scripts/prepare_verify.py` converts it for smoke tests. The full 600 and the
  stage-matching evaluation need to come from the authors.
- **MMPerspective:** on the Hub as `yoloytang/MMPerspective` (images in `root_v3/<task>/`,
  annotations expected as JSON). Run `python scripts/prepare_mmperspective.py --inspect`
  first to confirm the layout, then without `--inspect` to write `data/mmperspective.jsonl`.
