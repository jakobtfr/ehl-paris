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
several diverse, valid room arrangements.

**Backend provenance:** the app shows whether it is running the baseline
fallback or a `FLOORGEN_CHECKPOINT` flow sampler, including checkpoint path,
device, sampler steps, and presence threshold. The GeoJSON output also records
the backend metadata for the displayed sample.

**Deploy:** create a Gradio Space, push this repo. Spaces reads this file's
front-matter; rename it to `README.md` in the Space (the repo's own README is
the project README).
