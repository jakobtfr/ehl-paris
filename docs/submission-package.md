# Judge Methodology Package

This package is the judge-facing substitute for a final slide deck when a deck
file is not present. It explains what is implemented, what is still blocked by
local artifacts, and how to verify the repository quickly.

## Method Summary

`floorgen` solves the challenge as vector generation, not image generation. The
generator samples fixed room slots represented as minimum rotated rectangles:
center, width, height, angle, type, and presence. The condition is the apartment
outline, encoded as sampled boundary points plus real-scale features such as
area, bounding-box size, perimeter, compactness, and aspect ratio.

The model path is flow matching. The repository contains both the original MLP
room-flow model and a transformer room-flow architecture that can load
checkpoints whose config declares `architecture = "transformer"`. The public
entry point remains `generate(outline)`.

The deterministic post-process is a validity repair layer. It clips generated
room rectangles to the outline, removes overlaps largest-first, and fills only
small uncovered slivers. Large overlaps or gaps are rejected and retried. Empty
repaired outputs are also retried, so checkpoint-backed generation is not
silently accepted as a zero-room layout.

## Architecture Map

```text
outline polygon
  -> sampled boundary + scale features
  -> flow model sampler
  -> MRR room slots and room-type logits
  -> deterministic repair/reject layer
  -> labelled vector room polygons
  -> MSD-style renderer and export/evaluation CLIs
```

Key files:

- `src/floorgen/generate.py` — evaluator entry point, backend loading, retries,
  backend provenance.
- `src/floorgen/model/network.py` — MLP and transformer flow models.
- `src/floorgen/model/sampler.py` — Euler sampler, checkpoint loader, non-empty
  presence-slot selection.
- `src/floorgen/repr/mrr.py` — MRR representation and validity repair.
- `src/floorgen/data/preprocess.py` — official train/test split support,
  leakage-safe train/val split from train only, unit/plan/floor metadata.
- `scripts/export_batch.py` — MSD `geom` WKT export with label and metadata.
- `scripts/evaluate.py` — generated validity report and optional real-vs-
  generated FID/PRDC report.
- `src/floorgen/demo/app.py` — Gradio demo with backend provenance.
- `docs/artifact-manifest.md` — local artifact hashes, ignored-file status, and
  regeneration commands.

## Challenge Alignment

| Challenge requirement | Implementation status |
| --- | --- |
| Input is apartment outline only | `generate(outline)` accepts Shapely polygons. |
| Output is typed vector room polygons | Room records include `label`, `label_idx`, `polygon`, and `geojson`; exports include WKT `geom`. |
| Diffusion/flow model, not pure solver | Flow model path and checkpoint loader exist; baseline is documented as fallback only. |
| Preserve outline and containment | Repair and validity metrics check outside, overlap, gap, invalid rate. |
| FID/density/coverage | Real-vs-generated rendered image metric path is implemented; blocked status is reported when dependencies/data are missing. |
| Kaggle split integrity | Preprocessing accepts explicit train/test CSVs or Kaggle floor-id marker folders, keeps test as `split="test"`, and derives val only from official train. |
| Process history | `origin/entire/checkpoints/v1` exists. |

## Verification Commands

```bash
uv run python scripts/smoke_test.py
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/evaluate.py --demo --n-samples 2
uv run python scripts/export_batch.py --demo --format csv --output-dir /tmp/floorgen-export-smoke
```

When processed MSD units and the AMD checkpoint are available locally:

```bash
uv run --extra train python scripts/evaluate.py \
  --units data/processed/units.jsonl \
  --split test \
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --device cpu \
  --steps 4 \
  --threshold 0.5 \
  --mode ranked \
  --candidate-budget 4 \
  --n-samples 1 \
  --real-metrics \
  --output reports/final_test_metrics_smoke.json

uv run --extra train python scripts/export_batch.py \
  --units data/processed/units.jsonl \
  --split test \
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --device cpu \
  --steps 4 \
  --threshold 0.5 \
  --mode ranked \
  --candidate-budget 4 \
  --n-samples 1 \
  --format csv \
  --output-dir outputs/final_test_export
```

## Current Artifact Status

| Artifact | Status |
| --- | --- |
| Code path for trained flow/transformer generation | Implemented. |
| Primary MSD-trained checkpoint | Present locally at `checkpoints/flow-transformer-amd-862d422.pt`, ignored by git, SHA256 in `docs/artifact-manifest.md`. |
| Processed official split units | Present locally at `data/processed/units.jsonl`, ignored by git; split counts are train=13,499, val=2,418, test=2,734. |
| Generated test-split export | Limited 3-unit official test smoke exists locally under `outputs/final_test_export`; full 2,734-unit export has not been run. |
| Real FID/density/coverage values | 3-unit smoke reported FID `257.3317565917969`, density `0.0`, coverage `0.0`; this is a smoke, not a final leaderboard-quality estimate. |
| Demo | Local Gradio demo implemented; no live deployment URL is committed. |
| Pitch deck file | Markdown deck source exists at `docs/pitch-deck.md`; no rendered PDF/PPTX is committed. |

## Limitations To State Clearly

- The baseline backend is only for smoke testing and demo continuity. It is not
  the official scored generator unless explicitly submitted as an emergency
  fallback.
- The AMD checkpoint's raw type logits currently collapse toward `Balcony`.
  Ranked mode records an explicit semantic-calibration fallback when collapsed
  labels are detected, so generated geometry remains checkpoint-derived while
  semantic repair is auditable.
- MRRs cannot exactly reconstruct every irregular MSD room; the oracle gate is
  included to quantify this representation loss.
- The renderer uses the centralized MSD-style room palette documented in
  `src/floorgen/eval/render.py`; exact organiser wrapper details such as graph
  overlays remain outside this repository.
- Real metrics are only meaningful when real processed layouts and generated
  layouts are rendered through the same config.

## Suggested Demo Script

1. Open the Gradio app with `uv run --extra demo python app.py`.
2. Show the backend provenance line. If no checkpoint is loaded, say it is the
   baseline fallback.
3. Generate three samples for a preset outline and point to the unchanged
   boundary, labels, and GeoJSON vector output.
4. If a checkpoint is available, launch with `FLOORGEN_CHECKPOINT=...` and show
   the checkpoint path, sampler steps, and threshold in the UI.
5. Show the exported `geom` WKT schema and evaluation report commands.
