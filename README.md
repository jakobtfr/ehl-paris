# floorgen — Generative Floor-Plan Completion from an Apartment Outline

**Davis AI / TUM.ai Hackathon — "Mirror Mirror on the Wall"**

Given only an apartment **outline polygon**, generate a complete, plausible set
of typed interior **room polygons** (vector, never a pixel grid) using a
diffusion / flow-matching model. Scored on **FID**, **density**, and
**coverage** against real Swiss residential floor plans (Modified Swiss
Dwellings).

---

## Challenge Requirements → Implemented Features

| Requirement | Status | Implementation |
|---|---|---|
| Input: apartment outline polygon | Done | `generate(outline)` accepts any Shapely `Polygon`/`MultiPolygon` |
| Output: typed room polygons (vector) | Done | Returns `list[dict]` with `label`, `polygon`, `geojson` per room |
| Diffusion/flow-matching model | In progress | Model seam at `GENERATOR` in `generate.py`; baseline fallback active |
| MRR room representation (cx, cy, w, h, angle, type) | Done | `src/floorgen/repr/mrr.py` |
| Deterministic validity-repair layer | Done | Clips to outline, resolves overlaps, fills slivers |
| FID evaluation (TorchMetrics) | Done | `src/floorgen/eval/metrics.py::compute_fid` |
| Density & Coverage (PRDC) | Done | `src/floorgen/eval/prdc.py` (vendored clovaai) |
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
│ GENERATOR backend  │◀─────│ Trained flow model (when ready) │
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
  data/
    outline.py         outline construction (buffer 0.3 / union / -0.3)
    normalize.py       256/max-delta model-space norm + scale conditioning
    preprocess.py      CSV → per-unit records + leakage-safe split + report
  repr/
    mrr.py             MRR encode/decode + deterministic validity-repair layer
    boxes.py           axis-aligned fallback representation
    oracle_gate.py     MRR reconstruction gate (go/no-go before training)
  eval/
    render.py          MSD-parity rasteriser (torch-free)
    prdc.py            density/coverage (vendored clovaai PRDC, numpy-only)
    metrics.py         validity + distribution + FID metrics
  demo/
    app.py             Gradio demo UI
    presets.json       real MSD apartment outlines for the demo
tests/
  test_contract.py     contract tests (generate() guarantees)
  test_repr.py         MRR round-trip tests
  test_eval.py         evaluation pipeline tests
  test_export.py       export utilities tests
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
```

### Environment Variables

See [`.env.example`](.env.example) for all variables.

| Variable | Purpose | Required? |
|---|---|---|
| `MSD_CSV_PATH` | Path to the Modified Swiss Dwellings CSV | For preprocessing only |
| `FLOORGEN_PROCESSED` | Output dir for preprocessed data (default: `data/processed`) | No |
| `FLOORGEN_REPORTS` | Output dir for preprocessing reports (default: `reports`) | No |

### Run

```bash
# 1. Preprocess the MSD dataset (requires MSD_CSV_PATH)
uv run python -m floorgen.data.preprocess --out data/processed --reports reports

# 2. Oracle gate — verify MRR representation fidelity
uv run python -m floorgen.repr.oracle_gate --units data/processed/units.jsonl

# 3. Generate rooms for an outline (smoke test)
uv run python -B -c "from shapely.geometry import box; from floorgen.generate import generate; print(len(generate(box(0,0,10,8))), 'rooms')"

# 4. Tests + lint
uv run --extra dev pytest -q
uv run --extra dev ruff check src tests

# 5. Launch Gradio demo (requires: pip install gradio)
uv run python app.py
```

---

## Demo Flow

1. **Launch** the Gradio app: `uv run python app.py` (or deploy to HuggingFace Spaces).
2. **Select** a preset outline (real MSD apartments of varying size) or paste custom WKT.
3. **Choose** number of samples (1–6) and seed.
4. **Generate** — the app calls `sample_layouts()` and renders coloured room polygons.
5. **Inspect** the GeoJSON output panel for machine-readable room geometry.

The demo always uses whatever backend is wired into `GENERATOR`. Today this is
the heuristic baseline; once the trained flow model is registered, the demo
upgrades automatically with no UI changes.

---

## Known Limitations

- **Baseline fallback active.** The current `GENERATOR` uses a heuristic
  space-partitioning baseline (`baseline.py`). This satisfies the `generate()`
  contract and produces valid geometry, but does **not** constitute the scored
  diffusion/flow model. Generated layouts are structurally plausible but not
  learned from data.
- **MRR compression.** Minimum rotated rectangles cannot perfectly represent
  L-shaped or irregular rooms. The oracle gate quantifies this; corner-sequence
  tokens are a documented stretch goal.
- **No trained weights committed.** Model training runs on the AMD GPU box. The
  `GENERATOR` seam allows hot-swapping once a checkpoint is available.
- **Evaluation requires torch.** FID and PRDC run only with `--extra train`
  installed (GPU box). Geometry-validity metrics are always available.
- **MSD CSV not included.** The dataset is ~370k rows and is sourced from
  Kaggle. Point `MSD_CSV_PATH` to your local copy.

---

## Wiring in the Trained Model

Implement a sampler `model_sample(outline, rng) -> list[RoomMRR]` and register:

```python
import floorgen.generate
floorgen.generate.GENERATOR = model_sample
```

The repair layer, evaluator, contract tests, demo, and `generate(outline)`
signature all remain unchanged. Document the generation seed, candidate count,
ranking method, checkpoint, and config hash for submitted outputs.

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
- [x] Honest limitations documented
- [ ] Trained flow/diffusion model registered as `GENERATOR`
- [ ] Generated test-split outputs in MSD `geom` format
- [ ] Pitch deck

---

## Process History

This repository uses Entire for full session history. The checkpoint branch
`entire/checkpoints/v1` captures the working-session record as required by the
hackathon rules. This is advisory and does not count toward placement.

---

## License

Hackathon project — Davis AI / TUM.ai. Dataset: Modified Swiss Dwellings
(Kaggle, CC BY-SA 4.0).
