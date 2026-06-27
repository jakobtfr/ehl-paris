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
hackathon. Pick a real Swiss apartment outline or paste your own polygon, then
sample several diverse, valid arrangements.

The backend swaps to the trained flow-matching model automatically once
registered — the UI does not change.

**Deploy:** create a Gradio Space, push this repo. Spaces reads this file's
front-matter; rename it to `README.md` in the Space (the repo's own README is
the project README).
