# floorgen — generative floor-plan completion from an apartment outline

Davis AI / TUM.ai hackathon ("Mirror Mirror on the Wall"). Given only an
apartment **outline**, generate a complete, plausible set of typed interior
**room polygons** (vector, never a pixel grid), using a diffusion / flow-matching
model. Scored on **FID**, **density**, and **coverage** against real Swiss
residential floor plans (Modified Swiss Dwellings).

This repo holds the **data + representation + evaluation** pipeline and the
`generate(outline)` contract. The generative model plugs into the `GENERATOR`
seam in `generate.py`.

## Approach

The model emits room **geometry directly** as axis-aligned boxes
(`cx, cy, w, h, type, presence`); a deterministic layer only **repairs
validity** (clip to outline, resolve overlaps/gaps, drop slivers). The model —
not a rule — decides room count, type, and shape, which keeps us clear of the
"no rule-based partitioner" constraint while guaranteeing clean vector output.

An **oracle reconstruction gate** proves the box representation can reconstruct
real plans before any training (current dev result: mean IoU **0.56**, area
outside outline **~0%**, area error **2.5%**).

## Layout

```
src/floorgen/
  config.py            seed, geometry constants, room taxonomy, slot budget
  seeding.py           seed_everything(42)
  data/
    outline.py         outline construction (buffer 0.3 / union / -0.3); labels
    normalize.py       unit-area shape norm + absolute-scale conditioning scalars
    preprocess.py      CSV -> per-unit records + leakage-safe split + report
  repr/
    boxes.py           box encode/decode + deterministic validity-repair layer
    oracle_gate.py     box-reconstruction gate (go/no-go before training)
  eval/
    render.py          MSD-parity rasteriser (torch-free)
    prdc.py            density/coverage (vendored clovaai PRDC, numpy-only)
    metrics.py         validity + distribution metrics; FID (TorchMetrics, lazy)
  baseline.py          heuristic sampler — BASELINE/fallback ONLY, never scored
  generate.py          generate(outline) entry point + sample_layouts()
tests/                 contract + representation round-trip tests
```

## Install

```bash
uv sync                      # core (data + repr + eval-without-torch)
uv sync --extra train        # adds torch / torchmetrics for FID + training (GPU box)
uv sync --extra dev          # pytest / ruff / mypy
```

The MSD CSV is **not** committed. Point to it via `MSD_CSV_PATH`.

## Run

```bash
# 1. preprocess
MSD_CSV_PATH=/path/to/mds_V2_5.372k.csv python -m floorgen.data.preprocess \
    --out data/processed --reports reports

# 2. oracle gate (representation go/no-go)
python -m floorgen.repr.oracle_gate --units data/processed/units.jsonl

# 3. generate (baseline backend until the model is wired in)
python -c "from shapely.geometry import box; from floorgen.generate import generate; \
           print(len(generate(box(0,0,10,8))), 'rooms')"

# 4. tests + lint
pytest -q && ruff check src tests
```

## Wiring in the trained model

Implement a sampler `model_sample(outline, rng) -> list[RoomBox]` and set
`floorgen.generate.GENERATOR = model_sample`. The repair layer, evaluator,
contract tests, and `generate(outline)` signature stay unchanged. Everything is
seeded with **42** across data, sampling, and evaluation.

## Notes / open items

- The exact MSD `plot.py` wrapper (image size, palette, whether graph
  nodes/edges are drawn) is configurable in `eval/render.py::RenderConfig`;
  set to organiser parity once confirmed.
- Box representation absorbs some small / L-shaped rooms (dev retention ~84%);
  corner-sequence tokens are the documented stretch if higher fidelity is needed.
