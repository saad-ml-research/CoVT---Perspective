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

| Task | n | Majority | Qwen2.5-VL-7B (paper) | Qwen2.5-VL-7B | CoVT-7B-seg_depth_dino | Δ [paired 95% CI] | McNemar p |
|---|---|---|---|---|---|---|---|
| VPP | 156 | 40.4 | 35.3 | 38.5 [30.8, 45.5] | 32.1 [25.0, 39.1] | **−6.4** [−11.5, −1.9] | 0.021 |
| CLP | 123 | 27.6 | 29.3 | 21.1 [13.8, 28.5] | 17.9 [11.4, 25.2] | −3.3 [−8.9, +2.4] | 0.39 |
| VAP | 162 | 48.8 | 70.4 | 68.5 [61.1, 75.9] | 69.1 [61.7, 75.9] | +0.6 [−3.1, +4.3] | 1.00 |
| LDP | 285 | 32.6 | 73.7 | 68.8 [63.5, 74.4] | 68.4 [63.2, 74.0] | −0.4 [−3.9, +3.2] | 1.00 |
| PTR | 606 | 57.8 | 42.4 | 24.1 [20.8, 27.6] | 13.0 [10.6, 15.8] | **−11.1** [−13.9, −8.4] | <0.001 |
| LRR | 151 | 30.5 | 44.4 | 45.0 [37.1, 53.0] | 36.4 [28.5, 44.4] | **−8.6** [−15.2, −2.6] | 0.015 |
| OVR | 308 | 26.0 | 32.1 | 44.2 [38.6, 50.0] | 42.5 [37.0, 48.1] | −1.6 [−6.2, +3.2] | 0.59 |
| PTS | 213 | 32.9 | 28.6 | 26.3 [20.7, 31.9] | 20.2 [15.0, 25.8] | −6.1 [−11.7, −0.5] | 0.053 |
| VPC | 114 | 47.4 | 44.7 | 45.6 [36.8, 55.3]¹ | 45.6 [36.8, 55.3]¹ | 0.0 | 1.00 |
| **All (per question)** | 2118 | 31.0 | — | **40.2** [38.1, 42.3] | **34.9** [32.9, 37.0] | **−5.3** [−6.8, −3.8] | <0.001 |
| **Task mean** | | | 44.5 | 42.5 | 38.4 | −4.1 | |

Δ = CoVT − Qwen on the same questions; bold = significant at p < 0.05 (exact McNemar).

¹ Both models answer B on every VPC question. Prediction agreement between the two models: 80.5%.

### MMPerspective: which visual expert? (direct prompt)

Accuracy (%), with the paired difference from Qwen2.5-VL-7B in parentheses. * = significant at p < 0.05 (exact McNemar).

| Task | Qwen2.5-VL-7B | CoVT seg | CoVT depth | CoVT seg+depth+dino | CoVT seg+depth+dino+edge |
|---|---|---|---|---|---|
| VPP | 38.5 | 27.6 (−10.9*) | 32.7 (−5.8*) | 32.1 (−6.4*) | 30.1 (−8.3*) |
| CLP | 21.1 | 22.0 (+0.8) | 19.5 (−1.6) | 17.9 (−3.3) | 21.1 (0.0) |
| VAP | 68.5 | 67.3 (−1.2) | 60.5 (−8.0*) | 69.1 (+0.6) | 69.1 (+0.6) |
| LDP | 68.8 | 69.1 (+0.4) | 68.8 (0.0) | 68.4 (−0.4) | 68.8 (0.0) |
| PTR | 24.1 | 14.2 (−9.9*) | 19.5 (−4.6*) | 13.0 (−11.1*) | 18.0 (−6.1*) |
| LRR | 45.0 | 37.7 (−7.3*) | 35.1 (−9.9*) | 36.4 (−8.6*) | 42.4 (−2.6) |
| OVR | 44.2 | 38.6 (−5.5*) | 37.3 (−6.8*) | 42.5 (−1.6) | 33.8 (−10.4*) |
| PTS | 26.3 | 20.7 (−5.6*) | 25.4 (−0.9) | 20.2 (−6.1) | 27.7 (+1.4) |
| VPC | 45.6 | 45.6 (0.0) | 45.6 (0.0) | 45.6 (0.0) | 45.6 (0.0) |
| **All** | **40.2** | **34.7 (−5.5*)** | **35.9 (−4.2*)** | **34.9 (−5.3*)** | **36.3 (−3.9*)** |

### MMPerspective: prompt sensitivity

Same image, question and options; only the wording around them changes.

**direct** (our format):
```
Question: Which line highlighted in the image is the Horizon Line?
Options:
A. Line A
B. Line B
C. Line C
D. Line D
Answer with the option letter only. Put the letter inside <answer></answer>.
```

**native** (the benchmark's own `prompt` field, verbatim; free-text answer, letter extracted afterwards):
```
Which line highlighted in the image is the Horizon Line? Select from the following choices. (A) Line A (B) Line B (C) Line C (D) Line D
```

| Task | Qwen direct | Qwen native | CoVT-sdd direct | CoVT-sdd native | Δ direct | Δ native |
|---|---|---|---|---|---|---|
| VPP | 38.5 | 34.6 | 32.1 | 36.5 | −6.4* | +1.9 |
| CLP | 21.1 | 35.8 | 17.9 | 24.4 | −3.3 | −11.4* |
| VAP | 68.5 | 69.8 | 69.1 | 69.1 | +0.6 | −0.6 |
| LDP | 68.8 | 60.0 | 68.4 | 76.8 | −0.4 | **+16.8*** |
| PTR | 24.1 | 26.2 | 13.0 | 23.6 | −11.1* | −2.6* |
| LRR | 45.0 | 46.4 | 36.4 | 35.1 | −8.6* | −11.3 |
| OVR | 44.2 | 36.0 | 42.5 | 31.8 | −1.6 | −4.2 |
| PTS | 26.3 | 31.5 | 20.2 | 25.4 | −6.1 | −6.1 |
| VPC | 45.6 | 39.5 | 45.6 | 45.6 | 0.0 | +6.1 |
| **All** | **40.2** | **39.4** | **34.9** | **38.6** | **−5.3*** | **−0.8** |

Prediction agreement between Qwen and CoVT-sdd: 80.5% (direct), 59.1% (native).

### MMPerspective: effects that hold under both prompts

CoVT-sdd − Qwen2.5-VL-7B (paired). * = p < 0.05.

| Task | Δ direct | Δ native | Consistent? |
|---|---|---|---|
| PTR | −11.1* | −2.6* | worse under both |
| LRR | −8.6* | −11.3 (p = 0.06) | worse under both |
| PTS | −6.1 (p = 0.05) | −6.1 | worse under both (n.s.) |
| VAP | +0.6 | −0.6 | no effect under either |
| VPC | 0.0 | +6.1 | no effect under either |
| LDP | −0.4 | +16.8* | prompt-dependent |
| CLP | −3.3 | −11.4* | prompt-dependent |
| VPP | −6.4* | +1.9 | prompt-dependent |
| OVR | −1.6 | −4.2 | no significant effect |
| **All** | **−5.3*** | **−0.8** | prompt-dependent |

### MMPerspective: switching CoVT's visual thoughts off and on

Same model (CoVT-7B-seg_depth_dino), same questions. *auto* = the model decides whether to emit visual-thought tokens; *off* = reply pre-filled with an empty thought block (`<think></think><answer>`); *on* = reply pre-filled with `<think>`, so every answer goes through visual thoughts. Accuracy (%); * = paired difference from *auto* significant at p < 0.05 (exact McNemar).

| Task | Qwen direct | auto | off | on | Qwen native | auto | off | on |
|---|---|---|---|---|---|---|---|---|
| VPP | 38.5 | 32.1 | 31.5 | 32.7 | 34.6 | 36.5 | 36.5 | 35.9 |
| CLP | 21.1 | 17.9 | 22.0 | 20.3 | 35.8 | 24.4 | 26.0 | 24.4 |
| VAP | 68.5 | 69.1 | 67.9 | 68.5 | 69.8 | 69.1 | 67.2 | 68.5 |
| LDP | 68.8 | 68.4 | 50.9* | 64.9* | 60.0 | 76.8 | 77.5 | 77.9 |
| PTR | 24.1 | 13.0 | 21.1* | 13.4 | 26.2 | 23.6 | 23.9 | 24.4 |
| LRR | 45.0 | 36.4 | 32.4 | 36.4 | 46.4 | 35.1 | 35.8 | 35.8 |
| OVR | 44.2 | 42.5 | 29.2* | 41.2 | 36.0 | 31.8 | 33.7 | 32.8 |
| PTS | 26.3 | 20.2 | 25.4 | 19.7 | 31.5 | 25.4 | 28.7 | 27.2 |
| VPC | 45.6 | 45.6 | 45.6 | 45.6 | 39.5 | 45.6 | 43.8 | 45.6 |
| **All** | **40.2** | **34.9** | **33.2** | **34.4** | **39.4** | **38.6** | **39.3** | **39.3*** |

Share of questions where CoVT emits visual thoughts under *auto*: 75.2% (direct), 82.0% (native).
Questions whose correctness changes relative to *auto*: off 355 / 2118 (direct), 57 / 2118 (native); on 30 / 2118 (direct), 28 / 2118 (native).
CoVT with thoughts *off* vs Qwen: −6.9* (direct), 0.0 (native; LDP +17.5*, CLP −9.8, LRR −10.6, PTR −2.3*).

PTR at Qwen's default minimum resolution (no upscaling): Qwen 22.8 (direct), 25.1 (native), vs 24.1 / 26.2 with CoVT's setting. The gap to the paper's 42.4 is not explained by resolution.

The edge expert's token is `<|pidinet_pad|>` (PiDiNet edge maps); CoVT-seg_depth_dino_edge emits it on 77% of questions (89% on PTR).
