# Results log

All runs on RunPod L40S 48 GB, transformers 4.50.1, torch 2.5, VLMEvalKit from
Wakals/CoVT @ c3497a3, exact-match answer extraction (no LLM judge).
Detailed tables come from `scripts/cvbench_breakdown.py` (bootstrap 95% CIs).

## Step 0: CV-Bench reproduction (2026-10-29)

### Summary

Official CV-Bench convention (VLMEvalKit / Cambrian-1): 2D accuracy is the mean of the
ADE20K and COCO accuracies; CV-Bench overall is the mean of 2D and 3D.

| Model | 2D | 3D | **CV-Bench** | Paper | Count | Depth | Distance |
|---|---|---|---|---|---|---|---|
| CoVT-7B-seg_depth_dino | 75.0 | 84.6 | **79.8** | 80.0 | 65.1 (paper 66.2) | 86.7 (paper 86.8) | 82.5 (paper 82.5) |
| Qwen2.5-VL-7B-Instruct | 75.2 | 73.1 | **74.1** | 74.5 | *pending* (paper 65.0) | 70.8 (paper 72.8) | 75.3 (paper 75.5) |
| **CoVT gain** | **-0.2** | **+11.5** | **+5.7** | +5.5 | | **+15.9** | **+7.2** |

Reproduction confirmed for both models: overall within 0.2 (CoVT) and 0.4 (baseline) of the
paper, and the CoVT gain (+5.7) matches the paper's +5.5. Baseline depth is 2.0 below the paper,
within its 95% CI (about ±3.6 at n = 600).

Averaging convention matters: pooling all 1,438 2D questions gives 75.9 (not 75.0), and a
pooled overall of 80.2. Always report the official source-averaged numbers when comparing
with the paper.

### CoVT-7B-seg_depth_dino: CV-Bench-2D (n = 1,438)

| Split | n | Acc | 95% CI |
|---|---|---|---|
| Count | 788 | 65.1 | [61.8, 68.3] |
| Relation | 650 | 88.9 | [86.5, 91.4] |
| ADE20K | 633 | 67.8 | [64.3, 71.4] |
| COCO | 805 | 82.2 | [79.6, 84.8] |
| Count / ADE20K | 342 | 52.6 | [47.4, 57.9] |
| Count / COCO | 446 | 74.7 | [70.6, 78.9] |
| Relation / ADE20K | 291 | 85.6 | [81.4, 89.3] |
| Relation / COCO | 359 | 91.6 | [88.9, 94.4] |

### CoVT-7B-seg_depth_dino: CV-Bench-3D (n = 1,200, all Omni3D)

| Split | n | Acc | 95% CI |
|---|---|---|---|
| Depth | 600 | 86.7 | [84.0, 89.3] |
| Distance | 600 | 82.5 | [79.5, 85.5] |
| Overall | 1200 | 84.6 | [82.6, 86.7] |

### Qwen2.5-VL-7B-Instruct (baseline): CV-Bench-2D (n = 1,438)

| Split | Acc |
|---|---|
| ADE20K | 68.2 |
| COCO | 82.1 |
| 2D (source mean) | 75.2 |

Per-task (Count / Relation) split pending `cvbench_breakdown.py`.

### Qwen2.5-VL-7B-Instruct (baseline): CV-Bench-3D (n = 1,200, all Omni3D)

| Split | Acc |
|---|---|
| Depth | 70.8 |
| Distance | 75.3 |
| Overall | 73.1 |

### CoVT vs. baseline

| Split | Baseline | CoVT | Gain |
|---|---|---|---|
| 2D: ADE20K | 68.2 | 67.8 | -0.4 |
| 2D: COCO | 82.1 | 82.2 | +0.1 |
| 3D: Depth | 70.8 | 86.7 | **+15.9** |
| 3D: Distance | 75.3 | 82.5 | **+7.2** |
| CV-Bench overall | 74.1 | 79.8 | **+5.7** |

### Observations

- **Counting is the weak spot**, especially on ADE20K scenes (52.6 vs 74.7 on COCO): cluttered
  indoor scenes with many small objects. Relations are near ceiling (88.9).
- **No 2D gain from CoVT**: baseline 75.2 vs CoVT 75.0 (ADE20K 68.2 vs 67.8, COCO 82.1 vs 82.2),
  consistent with the paper's small Count gain (65.0 to 66.2).
- **The whole CoVT gain is 3D**: +15.9 on depth and +7.2 on distance, against roughly zero on 2D.
  Visual thoughts help specific geometric perception (depth ordering), not perception in general.
  This is the motivating pattern for the per-task MMPerspective analysis.
- **3D is strong** (84.6), consistent with CoVT's depth tokens being its main gain.
- Run notes: 3D ran with SDPA attention (flash-attn installed mid-run); 2D ran with
  flash-attn after an SDPA out-of-memory failure on large images.
