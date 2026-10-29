# Visual Thought Tokens: Do They Help MLLMs See, Reason, or Both?

## Results

### CV-Bench reproduction

Overall = mean of 2D and 3D accuracy.

| Model | Count | Relation | 2D | Depth | Distance | 3D | **Overall** | Paper |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 66.0 | 88.2 | 76.0 | 70.8 | 75.3 | 73.1 | **74.5** | 74.5 |
| CoVT-7B-seg_depth_dino | 65.1 | 88.9 | 75.9 | 86.7 | 82.5 | 84.6 | **80.2** | 80.0 |
| Δ (CoVT − base) | −0.9 | +0.7 | −0.1 | **+15.9** | **+7.2** | **+11.5** | **+5.7** | +5.5 |

### CV-Bench by task and image source (accuracy, 95% CI)

| Split | n | Qwen2.5-VL-7B | CoVT-7B-seg_depth_dino |
|---|---|---|---|
| Count / ADE20K | 342 | 52.9 [47.7, 57.9] | 52.6 [47.4, 57.9] |
| Count / COCO | 446 | 76.0 [72.0, 80.0] | 74.7 [70.6, 78.9] |
| Relation / ADE20K | 291 | 86.3 [82.1, 90.4] | 85.6 [81.4, 89.3] |
| Relation / COCO | 359 | 89.7 [86.6, 92.8] | 91.6 [88.9, 94.4] |
| Depth / Omni3D | 600 | 70.8 [67.3, 74.5] | 86.7 [84.0, 89.3] |
| Distance / Omni3D | 600 | 75.3 [72.0, 78.7] | 82.5 [79.5, 85.5] |
