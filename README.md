# floorgen — Generative Floor-Plan Completion from an Apartment Outline

**Davis AI / TUM.ai Hackathon — "Mirror Mirror on the Wall"**

Given only an apartment **outline polygon**, generate a complete, plausible set
of typed interior **room polygons** (vector, never a pixel grid) using a
diffusion / flow-matching model. Scored on **FID**, **density**, and
**coverage** against real world Swiss residential floor plans (Modified Swiss
Dwellings).

---

## Challenge Requirements → Implemented Features

| Requirement | Status | Implementation |
|---|---|---|
| Input: apartment outline polygon | Done | `generate(outline)` accepts any Shapely `Polygon`/`MultiPolygon` |
| Output: typed room polygons (vector) | Done | Returns `list[dict]` with `label`, `polygon`, `geojson` per room |
| Diffusion/flow-matching model | Training path implemented | `src/floorgen/model/*` + `scripts/train_flow.py`; baseline fallback remains active until a checkpoint is registered |
| MRR room representation (cx, cy, w, h, angle, type) | Done | `src/floorgen/repr/mrr.py` |
| Deterministic validity-repair layer | Done | Clips to outline, resolves overlaps, fills slivers |
| FID evaluation (TorchMetrics) | Implemented | `scripts/evaluate.py --units ... --real-metrics`; reports concrete blockers if deps/data are unavailable |
| Density & Coverage (PRDC) | Implemented | `src/floorgen/eval/prdc.py` + rendered real-vs-generated image report |
| MSD-parity rasteriser | Done | `src/floorgen/eval/render.py` |
| Data preprocessing pipeline | Done | `src/floorgen/data/preprocess.py` — CSV → per-unit records |
| Oracle reconstruction gate | Done | `src/floorgen/repr/oracle_gate.py` — MRR fidelity go/no-go |
| Reproducibility (seed 42) | Done | Global seed in `config.py`, per-sample RNG |
| Live demo | Done | Gradio app (`app.py` / HuggingFace Spaces) |
| Process history (Entire) | Done | `entire/checkpoints/v1` branch |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        generate(outline)                       │
│  Entry point called by evaluator. Returns room records.       │
└────────┬──────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐      ┌────────────────────────────────┐
│ GENERATOR backend  │◀─────│ Trained flow checkpoint          │
│ (pluggable seam)   │      │ OR baseline heuristic (default) │
└────────┬───────────┘      └────────────────────────────────┘
         │ list[RoomMRR]
         ▼
┌────────────────────────────────────────────────────────────────┐
│         Validity-Repair Layer (repr/mrr.py)                    │
│  clip to outline → resolve overlaps → fill slivers → validate │
└────────┬───────────────────────────────────────────────────────┘
         │ list[tuple[Polygon, int]]
         ▼
┌────────────────────────────────────────────────────────────────┐
│  Output: room records [{label, label_idx, polygon, geojson}]   │
└────────────────────────────────────────────────────────────────┘

Data Pipeline:
  MSD CSV → preprocess.py → per-unit JSONL → normalize.py → model-space tensors

Evaluation:
  generate() → render.py (rasterise) → metrics.py (FID, density, coverage)
```

### Module Map

```
src/floorgen/
  config.py            seed, geometry constants, room taxonomy, slot budget
  seeding.py           seed_everything(42)
  generate.py          generate(outline) entry + sample_layouts()
  baseline.py          heuristic sampler — fallback ONLY, never scored
  export.py            batch export utilities
  posttrain.py         checkpoint registration, scoring, export, provenance
  data/
    outline.py         outline construction (buffer 0.3 / union / -0.3)
    normalize.py       256/max-delta model-space norm + scale conditioning
    preprocess.py      CSV → per-unit records + leakage-safe split + report
  repr/
    mrr.py             MRR encode/decode + deterministic validity-repair layer
    boxes.py           axis-aligned fallback representation
    oracle_gate.py     MRR reconstruction gate (go/no-go before training)
    README.md          representation contract documentation
  model/
    data.py            processed JSONL → fixed-slot model tensors
    geometry.py        outline conditioning + MRR tensor encode/decode
    matching.py        Hungarian room-slot matching
    network.py         MLP + transformer fixed-slot room flow networks
    losses.py          flow, presence, and room-type training losses
    sampler.py         Euler sampler + checkpoint loader for GENERATOR
  eval/
    render.py          MSD-parity rasteriser (torch-free)
    prdc.py            density/coverage (vendored clovaai PRDC, numpy-only)
    metrics.py         validity + distribution + FID metrics
    realism.py         real-vs-generated FID/PRDC report helper
  demo/
    app.py             Gradio demo UI
    presets.json       real MSD apartment outlines for the demo
scripts/
  train_flow.py        train/smoke-test the conditional flow model
  post_train.py        checkpoint → score → export → report pipeline
  smoke_test.py        standalone pipeline smoke test (no deps beyond core)
  evaluate.py          full evaluation CLI (generate → validate → render → metrics)
  export_batch.py      batch export generated layouts to Parquet/CSV, optional checkpoint
tests/
  test_contract.py     contract tests (generate() guarantees)
  test_repr.py         MRR round-trip tests
  test_repair.py       validity-repair layer tests
  test_eval.py         evaluation pipeline tests
  test_export.py       export utilities tests
  test_model_*.py      flow-model data/loss/sampler/train smoke tests
```

---

## Setup & Run

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
# Core (data + repr + eval-without-torch)
uv sync

# Dev (pytest + ruff + mypy)
uv sync --extra dev

# Full (adds torch / torchmetrics for FID + training — GPU box)
uv sync --extra train

# Demo UI (adds Gradio)
uv sync --extra demo
```

### Environment Variables

See [`.env.example`](.env.example) for all variables.

| Variable | Purpose | Required? |
|---|---|---|
| `MSD_CSV_PATH` | Path to one Modified Swiss Dwellings CSV; dev fallback when official split files are not passed | For single-CSV preprocessing only |
| `MSD_TRAIN_CSV_PATH` | Path to Kaggle's predefined train split CSV | For official-split preprocessing |
| `MSD_TEST_CSV_PATH` | Path to Kaggle's predefined test split CSV | For official-split preprocessing |
| `MSD_KAGGLE_DIR` | Directory containing one train CSV and one test CSV; auto-detected by filename | Alternative to explicit split paths |
| `FLOORGEN_PROCESSED` | Output dir for preprocessed data (default: `data/processed`) | No |
| `FLOORGEN_REPORTS` | Output dir for preprocessing reports (default: `reports`) | No |
| `FLOORGEN_CHECKPOINT` | Optional checkpoint path auto-loaded by `floorgen.generate` | For trained generation |
| `FLOORGEN_DEVICE` | Torch device for checkpoint generation (`cpu`, `cuda`) | No |
| `FLOORGEN_SAMPLE_STEPS` | Euler steps for checkpoint sampler | No |
| `FLOORGEN_PRESENCE_THRESHOLD` | Room-presence threshold for checkpoint sampler | No |

### Run

```bash
# 1. Preprocess the official Kaggle train/test split.
#    Official test stays split=test; official train is split into train/val.
uv run python -m floorgen.data.preprocess \
  --train-csv "$MSD_TRAIN_CSV_PATH" \
  --test-csv "$MSD_TEST_CSV_PATH" \
  --out data/processed --reports reports

# Development fallback for a single CSV: creates a local train/val split only.
uv run python -m floorgen.data.preprocess \
  --csv "$MSD_CSV_PATH" \
  --out data/processed-dev --reports reports-dev

# 2. Oracle gate — verify MRR representation fidelity
uv run python -m floorgen.repr.oracle_gate --units data/processed/units.jsonl

# 3. Train-flow smoke test (requires preprocessed units.jsonl; CPU-friendly)
uv run --extra train python scripts/train_flow.py \
  --data data/processed/units.jsonl \
  --out checkpoints/flow-smoke.pt \
  --epochs 1 --batch-size 2 --max-steps 2 --device cpu

# 4. Generate rooms for an outline (baseline smoke test unless checkpoint registered)
uv run python -B -c "from shapely.geometry import box; from floorgen.generate import generate; print(len(generate(box(0,0,10,8))), 'rooms')"

# 5. Post-training checkpoint evaluation/export
uv run --extra train python scripts/post_train.py \
  --checkpoint checkpoints/flow-smoke.pt \
  --units data/processed/units.jsonl \
  --output-dir reports/post_train \
  --split test --n-samples 4 --steps 32 --threshold 0.5

# 5b. Real-vs-generated rendered FID/PRDC report, when processed units exist.
uv run --extra train python scripts/evaluate.py \
  --units data/processed/units.jsonl \
  --split test \
  --real-metrics \
  --n-samples 4 \
  --output reports/eval/test_real_metrics.json

# 6. Use a trained checkpoint through generate()
FLOORGEN_CHECKPOINT=checkpoints/flow-smoke.pt \
uv run --extra train python -B -c "from shapely.geometry import box; from floorgen.generate import generate; print(len(generate(box(0,0,10,8))), 'rooms')"

# 7. Tests + lint. Include --extra train to exercise model/post-training tests.
uv run --extra dev --extra train pytest -q
uv run --extra dev ruff check src tests

# 8. Quick smoke test (no external data or gradio needed)
uv run python scripts/smoke_test.py

# 9. Launch Gradio demo
uv run --extra demo python app.py
```

---

## Demo Flow

1. **Launch** the Gradio app: `uv run python app.py` (or deploy to HuggingFace Spaces).
2. **Select** a preset outline (real MSD apartments of varying size) or paste custom WKT.
3. **Choose** number of samples (1–6) and seed.
4. **Generate** — the app calls `sample_layouts()` and renders coloured room polygons.
5. **Inspect** the GeoJSON output panel for machine-readable room geometry.

The demo always uses whatever backend is wired into `GENERATOR`. It shows
backend provenance in the UI and records it in the GeoJSON output: baseline,
custom registered generator, or `FLOORGEN_CHECKPOINT` with device, steps, and
presence threshold.

---

## Known Limitations

- **Baseline fallback active by default.** The current `GENERATOR` uses a heuristic
  space-partitioning baseline (`baseline.py`). This satisfies the `generate()`
  contract and produces valid geometry, but does **not** constitute the scored
  diffusion/flow model unless replaced by a loaded checkpoint sampler.
- **Training code exists; trained weights are not committed in this checkout.**
  The fixed-slot MLP and transformer flow models, losses, sampler, and training
  script are implemented. A local checkpoint path can be loaded via
  `FLOORGEN_CHECKPOINT` or CLI `--checkpoint`, but this repository currently
  does not include a real MSD-trained `.pt` file.
- **MRR compression.** Minimum rotated rectangles cannot perfectly represent
  L-shaped or irregular rooms. The oracle gate quantifies this; corner-sequence
  tokens are a documented stretch goal.
- **Image metrics require torch.** FID and PRDC run only with `--extra train`
  installed. If dependencies, real processed units, or sample count are
  insufficient, `scripts/evaluate.py --real-metrics` writes an explicit
  `image_metrics.status = "blocked"` instead of fabricating scores.
- **MSD CSV not included.** The dataset is ~370k rows and is sourced from
  Kaggle. Point `MSD_CSV_PATH` to your local copy.

---

## Wiring in the Trained Model

Train a checkpoint with `scripts/train_flow.py`, load it as a
`GENERATOR`-compatible sampler, and register it:

```python
import floorgen.generate
from floorgen.model.sampler import load_generator

floorgen.generate.GENERATOR = load_generator(
    "checkpoints/flow.pt",
    device="cpu",      # or "cuda" on the GPU box
    steps=32,
    threshold=0.5,
)
```

The repair layer, evaluator, contract tests, demo, and `generate(outline)`
signature all remain unchanged. Document the generation seed, candidate count,
ranking method, checkpoint, and config hash for submitted outputs.

For the full post-training handoff, run `scripts/post_train.py`. It loads the
checkpoint, scores validation outlines, exports generated layouts, and writes:

- `reports/post_train/post_train_report.json`
- `reports/post_train/post_train_summary.md`
- `reports/post_train/layouts/layouts_*.parquet` or `.csv`

---

## Submission Checklist

- [x] `generate(outline)` contract — one positional arg, returns room records
- [x] Contract tests pass (`pytest -q`)
- [x] Linter clean (`ruff check src tests`)
- [x] Live demo (Gradio, deployable to HuggingFace Spaces)
- [x] Entire/checkpoint branch pushed (`entire/checkpoints/v1`)
- [x] Evaluation pipeline (FID, density, coverage)
- [x] Data pipeline (preprocessing, outline construction, normalization)
- [x] MRR representation with oracle validation gate
- [x] Flow-model training/sampling code path
- [x] Honest limitations documented
- [ ] Trained flow/diffusion model registered as `GENERATOR`
- [ ] Real MSD preprocessing artifacts (`data/processed/*`, reports)
- [ ] Real trained checkpoint + checkpoint metadata in this checkout
- [ ] Generated test-split outputs in MSD `geom` format
- [x] Judge-ready methodology package (`docs/submission-package.md`)
- [ ] Final pitch deck PDF/PPTX

---

## Process History

This repository uses Entire for full session history. The checkpoint branch
`entire/checkpoints/v1` captures the working-session record as required by the
hackathon rules. This is advisory and does not count toward placement. 

---

## License

Hackathon project — Davis AI / TUM.ai. Dataset: Modified Swiss Dwellings
(Kaggle, CC BY-SA 4.0).
