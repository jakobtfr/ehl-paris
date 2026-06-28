# Evaluation Pipeline

## Overview

The evaluation pipeline validates generated floor plans against ground-truth
outlines using geometry validity metrics, distribution analysis, and MSD-style
rendering for FID/PRDC scoring.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev --extra train

# Run evaluation on demo outlines (no data needed)
uv run python scripts/evaluate.py --demo --n-samples 4

# Run with trained model checkpoint
FLOORGEN_CHECKPOINT=checkpoints/flow-transformer-amd-862d422.pt \
FLOORGEN_GENERATION_MODE=ranked \
FLOORGEN_CANDIDATE_BUDGET=16 \
uv run --extra train python scripts/evaluate.py --demo --n-samples 1 --mode ranked

# Compute real-vs-generated FID/PRDC when processed units are available.
# Requires torch/torchmetrics; if unavailable, the JSON report records the blocker.
uv run --extra train python scripts/evaluate.py \
  --units data/processed/units.jsonl \
  --split test \
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --threshold 0.5 \
  --mode ranked \
  --real-metrics \
  --n-samples 1 \
  --output reports/final_test_metrics_smoke.json

# Batch export to Parquet
uv run python scripts/export_batch.py --demo --format parquet

# Batch export generated layouts for processed units.
uv run --extra train python scripts/export_batch.py \
  --units data/processed/units.jsonl \
  --split test \
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --threshold 0.5 \
  --mode ranked \
  --format csv \
  --output-dir outputs/final_test_export
```

## Architecture

```
outline geometry
       │
       ▼
┌─────────────┐
│  generate() │  ← baseline or trained flow transformer
└──────┬──────┘
       │ list[dict] with polygon, label, label_idx
       ▼
┌──────────────────┐
│ validity_metrics │  → outside_frac, overlap_frac, gap_frac, invalid_rate
└──────┬───────────┘
       │
       ▼
┌───────────────┐
│ render_layout │  → 512×512 uint8 image (MSD-style)
└──────┬────────┘
       │
       ▼
┌────────────────────────┐
│ FID / PRDC (optional)  │  ← requires torch + torchmetrics
└────────────────────────┘
```

## Metrics

### Geometry Validity (torch-free)

| Metric | Description | Ideal |
|--------|-------------|-------|
| `outside_frac` | Fraction of room area outside the outline | 0.0 |
| `overlap_frac` | Double-counted area between rooms / outline area | 0.0 |
| `gap_frac` | Uncovered outline area / outline area | 0.0 |
| `invalid_rate` | Fraction of rooms with invalid Shapely geometry | 0.0 |
| `n_rooms` | Number of rooms generated | 3–20 |

### Distribution Metrics (torch-free)

| Metric | Description |
|--------|-------------|
| `room_count_mean` | Average rooms per layout |
| `room_count_std` | Room count variance across layouts |
| `label_freq` | Per-label frequency distribution |

### Image Metrics (requires torch)

| Metric | Description |
|--------|-------------|
| FID | Fréchet Inception Distance vs real rendered layouts |
| Precision | Fraction of fakes inside real manifold |
| Recall | Fraction of reals covered by fake manifold |
| Density | How tightly fakes cluster near reals |
| Coverage | Fraction of real modes covered |

## Rendering

The renderer mirrors the MSD organiser's `plot.py` conventions:

- **Size**: 512×512 pixels
- **DPI**: 100
- **Background**: white
- **Rooms**: filled with room-type colour, 1px black edges
- **Aspect**: equal, centred in square frame
- **Output**: `(H, W, 3)` uint8 numpy array

### Room Colour Palette

| Room Type | RGB |
|-----------|-----|
| Bedroom | (135, 206, 235) |
| Livingroom | (255, 165, 0) |
| Kitchen | (220, 20, 60) |
| Dining | (255, 215, 0) |
| Corridor | (190, 190, 190) |
| Stairs | (139, 69, 19) |
| Storeroom | (160, 160, 200) |
| Bathroom | (60, 179, 113) |
| Balcony | (152, 251, 152) |
| Structure | (90, 90, 90) |

## Scoring

The `ScoreSummary` produces judge-friendly aggregate scores:

- **Validity Score (0–100)**: Penalises outside, overlap, gap, and invalid geometries
- **Overall Score (0–100)**: Weighted combination of validity (60%), success rate (25%), and label diversity (15%)

```python
from floorgen.eval.scoring import score_batch
from shapely.geometry import box

outlines = {"apt_1": box(0, 0, 12, 10)}
summary = score_batch(outlines, n_samples=4, seed=42)
print(summary.to_markdown())
```

## Batch Export

Exports generated layouts as structured data:

### Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `unit_id` | str | Source outline identifier |
| `sample_idx` | int | Sample number (0-based) |
| `seed` | int | Generation seed |
| `label` | str | Room type name |
| `label_idx` | int | Room type index |
| `wkt` | str | WKT polygon geometry |
| `area_m2` | float | Room area |
| `v_outside_frac` | float | Validity: outside fraction |
| `v_overlap_frac` | float | Validity: overlap fraction |
| `v_gap_frac` | float | Validity: gap fraction |
| `v_invalid_rate` | float | Validity: invalid geometry rate |
| `v_n_rooms` | int | Validity: room count |

### Metadata Sidecar (JSON)

Each export writes a `_meta.json` file with:
- Timestamp, seed, checkpoint name
- Number of outlines and samples
- Total rooms exported
- Column schema and room label taxonomy
- Generation failures, if `--allow-partial` was used

## Real-vs-Generated Metrics

`scripts/evaluate.py --units ... --real-metrics` loads processed `units.jsonl`
records, renders the real room polygons and generated room polygons through the
same `RenderConfig`, then attempts:

- TorchMetrics FID over rendered image stacks
- PRDC precision/recall/density/coverage over Inception features

The report writes an `image_metrics` object. When the heavy dependencies or
sample count are insufficient, the fields remain `null` and `status` is
`blocked` with the concrete exception. This is intentional: reports should show
metric blockers rather than invented scores.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOORGEN_CHECKPOINT` | _(none)_ | Path to `.pt` checkpoint file |
| `FLOORGEN_DEVICE` | `cpu` | Torch device (`cpu` or `cuda`) |
| `FLOORGEN_SAMPLE_STEPS` | `32` | Euler integration steps |
| `FLOORGEN_PRESENCE_THRESHOLD` | `0.5` | Room presence probability cutoff |
| `FLOORGEN_GENERATION_MODE` | `raw` | `raw` or `ranked` for `generate(outline)` |
| `FLOORGEN_CANDIDATE_BUDGET` | `16` | Candidate pool size for ranked mode |

## Running Tests

```bash
# All eval/export tests
uv run pytest tests/test_eval.py tests/test_export.py tests/test_eval_integration.py tests/test_eval_scoring.py -q

# Full suite
uv run pytest tests/ -q

# Lint
uv run ruff check src/floorgen/eval tests
```

## Current Results (Baseline)

```text
Outlines: 5 (demo rectangular)
Samples: 4 per outline
Total layouts: 20
Failures: 0

Validity:
  outside_frac_mean: 0.0000
  overlap_frac_mean: 0.0000
  gap_frac_mean:     0.0000
  invalid_rate_mean: 0.0000
  perfect_partitions: 20/20

Distribution:
  room_count_mean: 12.6
  room_count_std:  2.33
  labels used: 8/10
```

## Current Results (Official Test Smoke)

Command source: `reports/final_test_metrics_smoke.json`.

```text
Split: test
Outlines: 3 official test units
Checkpoint: checkpoints/flow-transformer-amd-862d422.pt
Mode: ranked
Candidate budget: 4
Samples: 1 per outline
Failures: 0

Validity:
  outside_frac_mean: 0.0000
  overlap_frac_mean: 0.0000
  gap_frac_mean:     0.0037
  invalid_rate_mean: 0.0000
  perfect_partitions: 3/3

Image metrics:
  FID:      257.3317565917969
  Density:  0.0
  Coverage: 0.0
```

## Known Limitations

- FID/PRDC require `torch` + `torchmetrics` (lazy import, fails gracefully)
- Raw checkpoint samples often produce overlapping rooms rejected by the strict repair layer
- Raw checkpoint type logits currently collapse toward `Balcony`; ranked mode records semantic calibration when it applies
- The real MSD checkpoint and processed data are ignored local artifacts; see
  `docs/artifact-manifest.md` for hashes and regeneration commands
- Demo outlines are simple rectangles; real MSD outlines have complex shapes
- Exact organiser wrapper details around graph overlays remain unverified
