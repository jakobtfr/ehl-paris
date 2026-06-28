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
| Diffusion/flow-matching model | Training + checkpoint path implemented | `src/floorgen/model/*` + `scripts/train_flow.py`; checkpoint-backed ranked mode is the judged path |
| MRR room representation (cx, cy, w, h, angle, type) | Done | `src/floorgen/repr/mrr.py` |
| Deterministic validity-repair layer | Done | Clips to outline, resolves overlaps, fills slivers |
| FID evaluation (TorchMetrics) | Implemented + smoke verified | `reports/final_test_metrics_smoke.json` reports numeric FID on 3 official test units |
| Density & Coverage (PRDC) | Implemented + smoke verified | `src/floorgen/eval/prdc.py`; smoke report has numeric density/coverage |
| MSD-parity rasteriser | Done | `src/floorgen/eval/render.py` |
| Data preprocessing pipeline | Done | `src/floorgen/data/preprocess.py` — CSV/Kaggle split folders → per-unit records |
| Oracle reconstruction gate | Done | `src/floorgen/repr/oracle_gate.py` — MRR fidelity go/no-go |
| Reproducibility (seed 42) | Done | Global seed in `config.py`, per-sample RNG |
| Local demo | Done | Gradio app (`app.py`); no deployed live URL is committed |
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
│ (pluggable seam)   │      │ OR baseline if checkpoint absent│
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
| `MSD_KAGGLE_DIR` | Extracted Kaggle directory; supports split CSVs or `train/full_out` + `test/full_out` floor-id markers | Alternative to explicit split paths |
| `FLOORGEN_PROCESSED` | Output dir for preprocessed data (default: `data/processed`) | No |
| `FLOORGEN_REPORTS` | Output dir for preprocessing reports (default: `reports`) | No |
| `FLOORGEN_MODEL` | Model alias when `FLOORGEN_CHECKPOINT` is unset: `amd-transformer` or `mlp`; default `amd-transformer` | No |
| `FLOORGEN_CHECKPOINT` | Optional checkpoint override; accepts aliases (`amd-transformer`, `mlp`) or a checkpoint path | No |
| `FLOORGEN_DEVICE` | Torch device for checkpoint generation (`auto`, `mps`, `cuda`, `cpu`); default `auto` prefers Mac MPS, then CUDA, then CPU | No |
| `FLOORGEN_SAMPLE_STEPS` | Euler steps for checkpoint sampler; default `16` | No |
| `FLOORGEN_PRESENCE_THRESHOLD` | Room-presence threshold for checkpoint sampler | No |
| `FLOORGEN_GENERATION_MODE` | `raw` or `ranked` for the one-argument `generate(outline)` entry point; default `ranked` | No |
| `FLOORGEN_CANDIDATE_BUDGET` | Candidate pool size for ranked generation; default `16` | No |

### Run

```bash
# 1. Preprocess the official Kaggle train/test split.
#    Official test stays split=test; official train is split into train/val.
uv run python -m floorgen.data.preprocess \
  --train-csv "$MSD_TRAIN_CSV_PATH" \
  --test-csv "$MSD_TEST_CSV_PATH" \
  --out data/processed --reports reports

# If the Kaggle archive has one geometry CSV plus split marker folders, use:
uv run python -m floorgen.data.preprocess \
  --kaggle-dir "$MSD_KAGGLE_DIR" \
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

# 4. Generate rooms for an outline.
#    If checkpoints/flow-transformer-amd-862d422.pt exists, this auto-loads
#    the AMD Transformer in ranked mode; otherwise it reports the fallback in
#    backend provenance.
uv run --extra train python -B -c "from shapely.geometry import box; import floorgen.generate as g; print(g.backend_provenance()); print(len(g.generate(box(0,0,10,8))), 'rooms')"

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
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --threshold 0.5 \
  --mode ranked \
  --real-metrics \
  --n-samples 1 \
  --output reports/final_test_metrics_smoke.json

# 6. Explicit AMD checkpoint command, equivalent to the local default when the
#    checkpoint exists.
uv run --extra train python -B -c "from shapely.geometry import box; import floorgen.generate as g; print(g.backend_provenance()); print(len(g.generate(box(0,0,10,8))), 'rooms')"

# 6b. Run the trained legacy MLP checkpoint locally on the same Mac GPU resolver.
FLOORGEN_MODEL=mlp \
FLOORGEN_DEVICE=auto \
uv run --extra train python -B -c "from shapely.geometry import box; import floorgen.generate as g; print(g.backend_provenance()); print(len(g.generate(box(0,0,10,8))), 'rooms')"

# 7. Tests + lint. Include --extra train to exercise model/post-training tests.
uv run --extra dev --extra train pytest -q
uv run --extra dev ruff check src tests

# 8. Quick smoke test (no external data or gradio needed; uses checkpoint inference)
uv run --extra train python scripts/smoke_test.py

# 9. Launch Gradio demo. The control panel is pre-filled with the AMD checkpoint
#    when checkpoints/flow-transformer-amd-862d422.pt exists locally.
uv run --with gradio --extra train python app.py

# 10. Launch checkpoint-backed judge dashboard with explicit env vars
FLOORGEN_CHECKPOINT=checkpoints/flow-transformer-amd-862d422.pt \
FLOORGEN_DEVICE=auto \
FLOORGEN_SAMPLE_STEPS=16 \
FLOORGEN_GENERATION_MODE=ranked \
FLOORGEN_CANDIDATE_BUDGET=16 \
uv run --with gradio --extra train python app.py
```

---

## Demo Flow

1. **Launch** the Gradio app: `uv run --with gradio --extra train python app.py`.
   The local default is the AMD Transformer checkpoint at
   `checkpoints/flow-transformer-amd-862d422.pt` in ranked mode.
2. **Select** a preset outline (real MSD apartments of varying size) or paste custom WKT.
3. **Choose** same-outline diversity, near-twin input sensitivity, raw-vs-ranked
   comparison, or single-sample inspection.
4. **Choose** raw samples or ranked/post-processed mode. Ranked mode passes the
   candidate budget into `sample_layouts(..., mode="ranked")` and records
   `floorgen.generate.LAST_RANKING_PROVENANCE`.
5. **Generate** 1-6 samples with a fixed seed; the app renders the vector room
   polygons and never uses a demo-only generation path.
6. **Read** the judge summary strip, which maps live evidence to FID realism,
   density, coverage, vector-output, audit trail, checkpoint, device, steps,
   threshold, candidate budget, and local report status.
7. **Inspect** Validity, Ranking Provenance, Vector Export, Run Provenance,
   Model, Metrics, Pitch Flow, and Limitations tabs. Vector export includes
   GeoJSON plus WKT/CSV rows and downloadable GeoJSON/CSV/provenance files.

The demo always uses whatever backend is wired into `GENERATOR`. In this local
workspace it auto-loads the AMD checkpoint when
`checkpoints/flow-transformer-amd-862d422.pt` exists. The UI still shows
baseline fallback, missing-checkpoint, and checkpoint-load-error states
explicitly for fresh clones or deployments where the large checkpoint artifact
has not been uploaded.

---

## Known Limitations

- **AMD checkpoint is the default local backend.** `floorgen.generate` now
  auto-loads `checkpoints/flow-transformer-amd-862d422.pt` in ranked mode when
  that file exists locally. The heuristic baseline (`baseline.py`) is retained
  only as a missing-artifact/debug fallback and should not be presented as the
  scored model.
- **AMD-trained checkpoint exists locally, but weights are not committed here.**
  The primary checkpoint path is `checkpoints/flow-transformer-amd-862d422.pt`.
  Raw strict repair still rejects many checkpoint samples because generated
  slots overlap too much. Ranked mode is documented test-time compute: it
  samples multiple candidates, applies repair-aware scoring, and selects
  diverse valid layouts. See `docs/artifact-manifest.md` for the local hash and
  artifact handoff status.
- **Checkpoint room-type head collapses in raw samples.** Direct type logits
  currently over-predict `Balcony`. Ranked mode records an explicit semantic
  calibration fallback when a candidate has collapsed labels, preserving model
  geometry while making the rendered semantic output usable. This is provenance,
  not a claim that raw type prediction is solved.
- **MRR compression.** Minimum rotated rectangles cannot perfectly represent
  L-shaped or irregular rooms. The oracle gate quantifies this; corner-sequence
  tokens are a documented stretch goal.
- **Image metrics require torch.** FID and PRDC run only with `--extra train`
  installed. If dependencies, real processed units, or sample count are
  insufficient, `scripts/evaluate.py --real-metrics` writes an explicit
  `image_metrics.status = "blocked"` instead of fabricating scores.
- **Large artifacts are ignored.** The Kaggle archive, processed units,
  checkpoints, reports, and exports are local ignored artifacts. See
  `docs/artifact-manifest.md` for hashes, status, and regeneration commands.

---

## Wiring in the Trained Model

The local runtime auto-loads `checkpoints/flow-transformer-amd-862d422.pt` when
that file is present. To train or test another checkpoint with
`scripts/train_flow.py`, load it as a `GENERATOR`-compatible sampler and
register it:

```python
import floorgen.generate
from floorgen.model.sampler import load_generator

floorgen.generate.GENERATOR = load_generator(
    "checkpoints/flow.pt",
    device="mps",      # use "cpu" only when local GPU is unavailable
    steps=16,
    threshold=0.5,
)
```

The repair layer, evaluator, contract tests, demo, and `generate(outline)`
signature all remain unchanged. Ranked generation is the default for the AMD
checkpoint path; raw mode is available for audits with
`FLOORGEN_GENERATION_MODE=raw`. Document the generation seed, candidate count,
ranking method, checkpoint, and config hash for submitted outputs.

For the full post-training handoff, run `scripts/post_train.py`. It loads the
checkpoint, scores validation outlines, exports generated layouts, and writes:

- `reports/post_train/post_train_report.json`
- `reports/post_train/post_train_summary.md`
- `reports/post_train/layouts/layouts_*.parquet` or `.csv`

---

## Submission Checklist

- [x] `generate(outline)` contract — one positional arg, returns room records
- [x] `generate(outline)` can use checkpoint-backed ranked mode via env vars
- [x] Contract tests pass (`pytest -q`)
- [x] Linter clean (`ruff check src tests`)
- [x] Local demo (Gradio, deployable to HuggingFace Spaces; no live URL committed)
- [x] Entire/checkpoint branch pushed (`entire/checkpoints/v1`)
- [x] Evaluation pipeline (FID, density, coverage)
- [x] Data pipeline (preprocessing, outline construction, normalization)
- [x] MRR representation with oracle validation gate
- [x] Flow-model training/sampling code path
- [x] Honest limitations documented
- [x] Trained flow/diffusion checkpoint load path (`FLOORGEN_CHECKPOINT`)
- [x] Real MSD preprocessing artifacts regenerated locally; ignored by git
- [x] Real trained checkpoint + checkpoint metadata available locally; ignored by git
- [x] Limited official test-split export smoke in MSD `geom` format
- [ ] Full 2,734-unit test-split export
- [x] Judge-ready methodology package (`docs/submission-package.md`)
- [x] Artifact manifest with local hashes (`docs/artifact-manifest.md`)
- [x] Markdown pitch deck (`docs/pitch-deck.md`)
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
