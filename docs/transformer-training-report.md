# Transformer Training Report

## Current Verified Status

The repository can now load transformer checkpoints through
`src/floorgen/model/sampler.py` when the checkpoint config contains
`architecture = "transformer"`. The architecture itself lives in
`src/floorgen/model/network.py::RoomFlowTransformer`.

This checkout does not contain a real MSD-trained checkpoint under
`checkpoints/`, so the trained-checkpoint probe cannot be completed locally in
this session. That is a data/artifact blocker, not a code-path blocker.

## Implemented Checkpoint Path

1. `FLOORGEN_CHECKPOINT` or CLI `--checkpoint` points to a `.pt` file.
2. `load_generator()` reads checkpoint `config`, rebuilds either
   `RoomFlowModel` or `RoomFlowTransformer`, loads `state_dict`, and returns a
   `GENERATOR`-compatible callable.
3. `sample_layouts()` calls the generator, runs deterministic repair, retries
   strict repair rejections, and now retries empty repaired outputs.
4. `generate(outline)` returns the first repaired layout with labelled vector
   polygons.

## Sampling Fixes Applied

- Empty repairs are no longer accepted as valid zero-room layouts.
- If presence thresholding keeps no room slots, the sampler keeps the highest
  model-ranked presence slot. This is still model-driven selection; it does not
  invoke the heuristic baseline or invent a partition.
- Export and evaluation CLIs can load checkpoints directly and write checkpoint
  SHA, sampler steps, threshold, device, and train metadata into reports.

## Local Probe Command

Run this when the checkpoint file exists:

```bash
uv run --extra train python - <<'PY'
from shapely.geometry import box, Polygon
import floorgen.generate as gen
from floorgen.model.sampler import load_generator
from floorgen.eval.metrics import validity_metrics

gen.GENERATOR = load_generator(
    "checkpoints/flow-transformer-862d422.pt",
    steps=64,
    threshold=0.5,
    device="cpu",
)

for name, outline in [
    ("rectangle_12x8", box(0, 0, 12, 8)),
    ("l_shape", Polygon([(0,0),(10,0),(10,6),(6,6),(6,10),(0,10)])),
]:
    layout = gen.sample_layouts(outline, seed=42, n_samples=1)[0]
    rooms = [(r["polygon"], r["label_idx"]) for r in layout]
    print(name, len(layout), validity_metrics(rooms, outline))
PY
```

## Remaining Artifact Blockers

- No checkpoint file is present in this checkout.
- No real processed MSD `units.jsonl` is present in this checkout.
- No generated test-split export artifact is present in this checkout.
- No real FID/density/coverage numbers should be claimed until the processed
  units and checkpoint are available and `scripts/evaluate.py --real-metrics`
  completes.
