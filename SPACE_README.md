---
title: floorgen — Floor Plan Generator
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# floorgen

Generate interior room layouts from an apartment outline. Davis AI / TUM.ai
hackathon ("Mirror Mirror on the Wall").

Pick a real Swiss apartment outline or paste your own polygon WKT, then sample
several diverse room arrangements. The dashboard shows same-outline diversity,
near-twin input sensitivity, raw-vs-ranked comparison, per-sample validity
metrics, ranking provenance, GeoJSON, WKT/CSV vectors, generation provenance,
model/checkpoint status, metric/report status, and a judge summary mapping the
live evidence to FID realism, density, coverage, vector-output, and audit-trail
criteria.

**Backend status:** the UI defaults to the AMD Transformer checkpoint at
`checkpoints/flow-transformer-amd-862d422.pt` and labels flow checkpoint
sampler, baseline fallback, custom generator, missing checkpoint, and
checkpoint-load errors explicitly. For a Space deployment, upload or mount the
checkpoint artifact at that path, or set `FLOORGEN_CHECKPOINT`,
`FLOORGEN_DEVICE`, `FLOORGEN_SAMPLE_STEPS`,
`FLOORGEN_PRESENCE_THRESHOLD`, `FLOORGEN_GENERATION_MODE`, and
`FLOORGEN_CANDIDATE_BUDGET` as Space secrets or runtime environment variables.
Use `FLOORGEN_MODEL=mlp` or `FLOORGEN_CHECKPOINT=mlp` to run the trained legacy
MLP checkpoint instead of the AMD Transformer when that artifact is present.
Set `FLOORGEN_GENERATION_MODE=ranked` for the judge path. Space requirements
include torch dependencies for checkpoint inference; without the checkpoint
artifact, the app can only show the missing-checkpoint/fallback state. GeoJSON,
CSV, and provenance downloads record the backend metadata for the displayed
sample.

**Deploy:** create a Gradio Space, push this repo. Spaces reads this file's
front-matter; rename it to `README.md` in the Space (the repo's own README is
the project README).
