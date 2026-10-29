# Results log

All runs on RunPod L40S 48 GB, transformers 4.50.1, torch 2.5, VLMEvalKit from
Wakals/CoVT @ c3497a3, exact-match answer extraction (no LLM judge).

## Step 0: CV-Bench reproduction (2026-10-29)

CV-Bench overall = mean of the 2D and 3D accuracies (Cambrian-1 convention).

| Model | 2D: ADE20K | 2D: COCO | 2D overall | 3D: Depth | 3D: Distance | 3D overall | **CV-Bench** | Paper |
|---|---|---|---|---|---|---|---|---|
| CoVT-7B-seg_depth_dino | 67.8 | 82.2 | 75.0 | 86.7 | 82.5 | 84.6 | **79.8** | 80.0 |
| Qwen2.5-VL-7B-Instruct | | | | | | | | 74.5 |

Notes:
- CoVT 3D ran with SDPA attention (flash-attn installed mid-run); 2D ran with flash-attn
  after an SDPA out-of-memory failure on large images. Depth and Distance match the paper
  (86.8, 82.5) to within 0.1.
- Reproduction confirmed: within 0.2 of the paper's 80.0.
