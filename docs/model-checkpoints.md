# Model Checkpoints

Date: 2026-06-28

This document is the quick judge-facing answer to: "What models did you train,
where are the weights, and why are there multiple checkpoints?"

## Short Answer

We have two real trained Transformer checkpoints plus two legacy/smoke
checkpoints:

- **Primary large Transformer:** `checkpoints/flow-transformer-amd-862d422.pt`
- **Local reproducible Transformer:** `checkpoints/flow-transformer-862d422.pt`
- **Transformer smoke checkpoint:** `checkpoints/flow-transformer-smoke.pt`
- **Legacy MLP checkpoints:** `checkpoints/flow-full-8303584.pt` and `checkpoints/flow-cpu-smoke.pt`

The Transformer checkpoints contain model weights and biases in `state_dict`.
They also include `optimizer_state_dict`, model config, room labels, and training
metadata. The legacy MLP checkpoints contain model weights and training metadata
but do not include optimizer state.

## Recommended Pitch

We did not stop at a single toy model. The checkpoint set shows the model
evolved through an inspectable progression:

1. **Legacy MLP baseline:** proves the fixed-slot MRR flow-matching pipeline can
   train and load.
2. **Transformer architecture:** replaces pooled MLP conditioning with
   outline-token encoding and slot-token decoding, matching the challenge need
   to reason over boundary geometry and room slots.
3. **Mac/MPS training run:** gives a locally reproducible trained Transformer
   checkpoint from the developer machine.
4. **AMD/ROCm training run:** scales the Transformer to a larger hidden size and
   batch size on stronger hardware.

This gives judges both a runnable fallback lineage and visible custom model
work. The honest limitation is that current raw samples still over-select slots
and overlap too much for the strict repair budget, so the next quality step is
presence calibration, overlap-aware loss/ranking, or validation-set model
selection.

## Checkpoint Inventory

| Checkpoint | Role | Architecture | Size | Device Recorded | Training | Initial Loss | Final Loss |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| `flow-transformer-amd-862d422.pt` | Primary large Transformer candidate | Transformer, `d_model=512`, 4 layers, 8 heads, FF=2048 | 352 MB | `cuda`, `ROCm device 0 (HIP 7.2.53211)` | 67 completed epochs, 4,212 steps, batch 256, 1 hour | 2856.5173 | 279.6894 |
| `flow-transformer-862d422.pt` | Local reproducible Transformer | Transformer, `d_model=256`, 4 layers, 8 heads, FF=1024 | 88 MB | `mps`, `Apple MPS` | 20 epochs, 4,960 steps, batch 64, 20m 16s | 2709.4360 | 228.6985 |
| `flow-transformer-smoke.pt` | Transformer smoke test | Transformer, `d_model=64`, 4 layers, 8 heads, FF=256 | 5.8 MB | `mps`, `Apple MPS` | 2 optimizer steps | 1970.9486 | 2606.6604 |
| `flow-full-8303584.pt` | Legacy MLP trained baseline | Legacy MLP, `d_model=128` | 618 KB | `cuda` | 10 epochs, 2,480 steps, batch 64 | 2698.7253 | 847.6898 |
| `flow-cpu-smoke.pt` | Legacy CPU smoke test | Legacy MLP, `d_model=32` | 54 KB | `cpu` | 2 optimizer steps | 2838.2876 | 2350.7759 |

The model-like `.pt` and `.pth` files under `.venv/` are third-party package
assets used by TorchMetrics. Other `.pth` files under `.venv/` are Python
environment path files. None of them are floorgen model checkpoints.

## Which Model Should We Show?

Use **`flow-transformer-amd-862d422.pt`** when the judge asks whether we used
accelerated training or stronger hardware. It is the largest model and records a
ROCm/HIP device, which corresponds to an AMD GPU stack exposed through PyTorch's
`cuda` device interface.

Use **`flow-transformer-862d422.pt`** when the judge asks for a fully local,
reproducible training run from the Mac. It has the best final training loss in
the checkpoint metadata, although training loss alone is not the challenge
score.

Use **the legacy MLP checkpoints only as baselines/ablation history**. They show
the pipeline existed before the Transformer upgrade, but they are not the model
we would pitch as the final challenge approach.

Both trained families can run locally on the Mac. Use `FLOORGEN_MODEL=mlp` (or
`FLOORGEN_CHECKPOINT=mlp`) to run `checkpoints/flow-full-8303584.pt` through the
same `FLOORGEN_DEVICE=auto` resolver, which prefers MPS on this machine.

## Architecture Summary

The current trainable model is `RoomFlowModel`, backed by
`RoomFlowTransformer` in `src/floorgen/model/network.py`.

The model consumes:

```python
model(x_t, t, outline_xy, scale, outline_mask) -> ModelOutput
```

It predicts:

- continuous room-token velocity for flow matching,
- room type logits,
- room presence logits.

Transformer-specific structure:

- outline points are embedded as boundary tokens,
- scale information is added as a conditioning token,
- an outline Transformer encoder builds memory over the apartment boundary,
- room slots are embedded as decoder tokens,
- a Transformer decoder predicts per-slot MRR geometry, type, and presence.

This is materially different from the old MLP, which pooled outline features and
ran slot-wise MLPs.

## Validation Snapshot

Both real Transformer checkpoints load through `load_generator` and generate
finite MRRs with positive dimensions for rectangle and L-shaped outlines.

Quick CPU sampling validation with 32 Euler steps:

| Checkpoint | Outline | Raw MRRs | Strict Repair | Permissive Repair |
| --- | --- | ---: | --- | --- |
| `flow-transformer-amd-862d422.pt` | Rectangle | 24 | rejected: overlap `1.733 > 0.250` | 6 valid rooms, outside 0.0, overlap 0.0, gap 0.0 |
| `flow-transformer-amd-862d422.pt` | L-shape | 24 | rejected: overlap `1.770 > 0.250` | 6 valid rooms, outside 0.0, overlap ~0.0, gap 0.0 |
| `flow-transformer-862d422.pt` | Rectangle | 24 | rejected: overlap `1.335 > 0.250` | 8 valid rooms, outside 0.0, overlap 0.0, gap 0.0 |
| `flow-transformer-862d422.pt` | L-shape | 24 | rejected: overlap `1.179 > 0.250` | 8 valid rooms, outside 0.0, overlap ~0.0, gap 0.0 |

Interpretation:

- The model checkpoints are real and numerically usable.
- The deterministic repair layer can turn permissively accepted outputs into
  valid polygons.
- Strict repair currently rejects useful samples because raw generated room
  slots overlap too much.
- The model still needs sampling calibration or overlap-aware validation before
  claiming final strict-validity quality.

## Ranked Post-Processing Snapshot

The AMD checkpoint is now wired through real ranked post-processing in
`sample_layouts(..., mode="ranked")`. Ranked mode generates a candidate pool,
strictly repairs when possible, falls back to documented permissive repair when
strict repair rejects a candidate, penalizes repair pressure, and selects
diverse layouts by stable signatures.

Representative CPU command:

```bash
uv run --extra train python scripts/evaluate.py \
  --demo \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --threshold 0.5 \
  --mode ranked --n-samples 2
```

Additional representative check on rectangle `12x8`, L-shape, and two real
demo presets wrote `reports/ranked_amd_representative.json`.

| Mode | Checkpoint | Outlines | Samples/outline | Candidate budget | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Raw strict repair | `flow-transformer-amd-862d422.pt` | 4 | 1 | n/a | 0/4 succeeded; all failed on large overlap repair |
| Ranked/post-processed | `flow-transformer-amd-862d422.pt` | 4 | 2 | 8 | 8/8 succeeded; outside mean 0.0, overlap mean 0.000051, gap mean 0.000286, invalid rate 0.0, mean 8.5 rooms |

This is a post-processing improvement over the raw checkpoint, not evidence
that the raw Transformer sampler is strict-valid. A follow-up audit found that
the AMD checkpoint's raw type logits collapse to `Balcony` on representative
test probes even though checkpoint `label_names` matches `ROOM_NAMES`. The
loader now rejects mismatched checkpoint label taxonomies, and ranked mode
records an explicit `semantic_repair` provenance entry when it applies an
area-ordered MSD semantic prior to collapsed candidates.

Verified one-argument ranked checkpoint probe:

```bash
FLOORGEN_CHECKPOINT=checkpoints/flow-transformer-amd-862d422.pt \
FLOORGEN_DEVICE=auto \
FLOORGEN_SAMPLE_STEPS=16 \
FLOORGEN_PRESENCE_THRESHOLD=0.5 \
FLOORGEN_GENERATION_MODE=ranked \
FLOORGEN_CANDIDATE_BUDGET=16 \
uv run --extra train python -B -c "from shapely.geometry import box; import floorgen.generate as g; layout = g.generate(box(0,0,10,8)); print(len(layout), sorted({r['label'] for r in layout}), g.LAST_RANKING_PROVENANCE.get('semantic_repair_count'))"
```

Observed local-MPS output with the current defaults: 10 rooms, labels
`Balcony`, `Bathroom`, `Bedroom`, `Corridor`, `Kitchen`, `Livingroom`,
`Structure`, and `semantic_repair_count=16`.

## Judge FAQ

**Were the weights and biases saved?**  
Yes. They are saved locally in each checkpoint's `state_dict`.

**Was anything logged to Weights & Biases?**  
No. "Weights and biases" here means the PyTorch parameters stored in the `.pt`
files, not the W&B SaaS product.

**Was the model trained on the Mac?**  
Yes. `flow-transformer-862d422.pt` was trained on Apple MPS for 20 epochs and
4,960 steps.

**Was the model trained on an AMD GPU?**  
Yes. `flow-transformer-amd-862d422.pt` records `ROCm device 0 (HIP 7.2.53211)`,
which is the AMD GPU software stack exposed through PyTorch's `cuda` device
label.

**Which checkpoint is the final model?**  
For the strongest hardware story, use `flow-transformer-amd-862d422.pt`. For
the best observed training-loss metadata, use `flow-transformer-862d422.pt`.
For a judged demo, select between them by sample quality on validation outlines,
not by file size alone.

**Are the legacy checkpoints still useful?**  
Yes, as baselines and process evidence. They should not be pitched as the final
model because the Transformer architecture is the intended challenge-aligned
implementation.

## Load Command

```python
from floorgen.model.sampler import load_generator

generate_mrrs = load_generator(
    "checkpoints/flow-transformer-amd-862d422.pt",
    steps=16,
    device="mps",
)
```

For demo use, point the app at a checkpoint:

```bash
FLOORGEN_CHECKPOINT=checkpoints/flow-transformer-amd-862d422.pt \
FLOORGEN_DEVICE=auto \
FLOORGEN_SAMPLE_STEPS=16 \
FLOORGEN_PRESENCE_THRESHOLD=0.5 \
FLOORGEN_GENERATION_MODE=ranked \
FLOORGEN_CANDIDATE_BUDGET=16 \
uv run --with gradio --extra train python app.py
```

The default `FLOORGEN_DEVICE=auto` prefers `mps` on the Mac, then `cuda`, then
`cpu`.

To run the trained MLP checkpoint locally on the Mac GPU:

```bash
FLOORGEN_MODEL=mlp \
FLOORGEN_DEVICE=auto \
uv run --extra train python -B -c "from shapely.geometry import box; import floorgen.generate as g; layout = g.generate(box(0,0,10,8)); print(g.backend_provenance()); print(len(layout), sorted({r['label'] for r in layout}))"
```
