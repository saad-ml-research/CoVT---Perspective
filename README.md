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

### MMPerspective (direct answer, no robustness split)

Accuracy (%) with bootstrap 95% CI. Paper = Qwen2.5-VL-7B as reported in MMPerspective (Tang et al., 2025). Majority = always answering the most common correct letter.

| Task | n | Majority | Qwen2.5-VL-7B (paper) | Qwen2.5-VL-7B | CoVT-7B-seg_depth_dino | Δ |
|---|---|---|---|---|---|---|
| VPP | 156 | 40.4 | 35.3 | 38.5 [30.8, 45.5] | 32.1 [25.0, 39.1] | −6.4 |
| CLP | 123 | 27.6 | 29.3 | 21.1 [13.8, 28.5] | 17.9 [11.4, 25.2] | −3.2 |
| VAP | 162 | 48.8 | 70.4 | 68.5 [61.1, 75.9] | 69.1 [61.7, 75.9] | +0.6 |
| LDP | 285 | 32.6 | 73.7 | 68.8 [63.5, 74.4] | 68.4 [63.2, 74.0] | −0.4 |
| PTR | 606 | 57.8 | 42.4 | 24.1 [20.8, 27.6] | 13.0 [10.6, 15.8] | −11.1 |
| LRR | 151 | 30.5 | 44.4 | 45.0 [37.1, 53.0] | 36.4 [28.5, 44.4] | −8.6 |
| OVR | 308 | 26.0 | 32.1 | 44.2 [38.6, 50.0] | 42.5 [37.0, 48.1] | −1.7 |
| PTS | 213 | 32.9 | 28.6 | 26.3 [20.7, 31.9] | 20.2 [15.0, 25.8] | −6.1 |
| VPC | 114 | 47.4 | 44.7 | 45.6 [36.8, 55.3]¹ | 45.6 [36.8, 55.3]¹ | 0.0 |
| **All (per question)** | 2118 | 31.0 | — | **40.2** [38.1, 42.3] | **34.9** [32.9, 37.0] | **−5.3** |
| **Task mean** | | | 44.5 | 42.5 | 38.4 | −4.1 |

¹ Both models answer B on every VPC question. Prediction agreement between the two models: 80.5%.
