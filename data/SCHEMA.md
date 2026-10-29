# Benchmark JSONL schema

Every benchmark is converted to one JSON object per line, so a single runner
(`covt_persp.infer`) and scorer (`covt_persp.score`) handle both.

```json
{"id": "mmp_vpp_0001",
 "task": "VPP",
 "images": ["images/mmp/vpp_0001.jpg"],
 "question": "Where is the vanishing point?",
 "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
 "answer": "B",
 "meta": {}}
```

- `images` paths are relative to the JSONL file's folder (or absolute). Paired
  tasks such as MMPerspective PTS list two images.
- `task`: MMPerspective task code (VPP, CLP, VAP, LDP, PTR, LRR, PTS, VPC, OVR,
  robustness) or VERIFY pattern type (GT, MA, GC, LP, EO, PR, SA, SC, SP).
- `meta`: anything extra, e.g. VERIFY's human reasoning path, MMPerspective's
  ground-truth vanishing point or image source (web / self-shot / Blender / RPVP).

Converters (`scripts/prepare_mmperspective.py`, `scripts/prepare_verify.py`)
come next, once the release format of each dataset is confirmed.
