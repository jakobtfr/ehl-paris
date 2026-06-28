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

**Backend status:** the UI explicitly labels baseline fallback, flow checkpoint
sampler, custom generator, missing checkpoint, and checkpoint-load errors. For
the checkpoint-backed demo, set `FLOORGEN_CHECKPOINT`,
`FLOORGEN_DEVICE`, `FLOORGEN_SAMPLE_STEPS`,
`FLOORGEN_PRESENCE_THRESHOLD`, and `FLOORGEN_CANDIDATE_BUDGET` as Space secrets
or runtime environment variables. GeoJSON, CSV, and provenance downloads record
the backend metadata for the displayed sample.

**Deploy:** create a Gradio Space, push this repo. Spaces reads this file's
front-matter; rename it to `README.md` in the Space (the repo's own README is
the project README).
