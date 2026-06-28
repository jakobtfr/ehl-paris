# Pitch Deck: floorgen
## From Apartment Outline to Plausible Rooms

**Team EHL Paris | Davis AI / TUM.ai Hackathon**

---

## Slide 1: The Challenge

**Given only an apartment outline, generate realistic interior room layouts.**

- Input: one boundary polygon (no rooms, no walls, no graph)
- Output: labelled vector room polygons that partition the outline
- Scored on: **FID** (realism), **Density** (on-manifold), **Coverage** (diversity)
- Against: real Swiss residential floor plans (Modified Swiss Dwellings)

> "Not one correct answer — a distribution of plausible plans."

---

## Slide 2: Why This Is Hard

**MSD is the most complex floor plan dataset available.**

| | RPLAN | LIFULL | MSD (ours) |
|---|---|---|---|
| Origin | Asia | Japan | Europe (Switzerland) |
| Corners/room | 4.54 | 5.04 | **8.68** |
| Rooms/unit | 6.67 | 8.15 | **8.75** |
| Units/floor | 1.00 | 1.00 | **3.52** |
| Non-Manhattan rooms | No | No | **Yes** |
| Graph entropy | 4.56 | 7.79 | **8.02** |

*Source: van Engelenburg et al., ECCV 2024, Table 1*

5,372 floor plans. 18.9K apartments. 165.3K labelled areas.
Higher diversity than any existing dataset.

---

## Slide 3: The Paper Insight We Used

**MRR tokens outperform full polygons for diffusion-based generation.**

From the MSD paper's own baseline (Modified HouseDiffusion):

| Representation | MIoU (avg) | Graph Compatibility |
|---|---|---|
| Full polygons (POL) + WCA | 17.9 | 71.1 |
| **MRR + WCA** | **21.8** | **76.2** |
| Full polygons (POL) | 10.9 | 80.3 |
| **MRR** | **11.5** | **87.1** |

MRR compresses each room into 5 numbers: `(cx, cy, w, h, angle)`.
Easier to learn. Preserves rotation. Stays vector-based.

We adopted this directly: our room token = **(center, size, angle, type, presence)**.

---

## Slide 4: Our Architecture

```
   Apartment Outline (Shapely Polygon)
              |
              v
   +------------------------------+
   | Boundary Encoder             |
   | 128 resampled points +       |
   | scale features (area, bbox)  |
   +------------------------------+
              |
              v
   +------------------------------+
   | Conditional Flow Model       |
   | Fixed K=24 room slots        |
   | Transformer, Euler sampling  |
   | Predicts: geometry + type +  |
   |           presence per slot   |
   +------------------------------+
              |
              v  MRR tokens (cx, cy, w, h, angle, type, presence)
   +------------------------------+
   | Validity Repair Layer        |
   | Clip to outline              |
   | Resolve overlaps             |
   | Fill slivers (<threshold)    |
   | Reject if repair too large   |
   +------------------------------+
              |
              v
   Labelled Room Polygons (vector output)
```

**Key principle:** The model decides room count, placement, shape, and type.
The repair layer only enforces the hard geometry contract.

---

## Slide 5: What Makes This Novel

1. **Outline-only conditioning** — no input graph, no target rooms, no rule-based partition. Harder than the MSD paper's setting (which uses both graph + structure).

2. **Flow matching over MRR tokens** — conditional flow model generates continuous room geometry directly. Not a GAN, not a pixel segmentation.

3. **Presence logits for variable room count** — fixed K slots with learned presence handles 4-room studios and 20-room penthouses in the same architecture.

4. **Strict generative/repair separation** — the model is generative; the repair is deterministic. We never let the repair layer *design* the floor plan.

---

## Slide 6: Evaluation-Aligned Design

| Judging Signal | Our System Response |
|---|---|
| **FID** | MSD-style 512x512 rasterization, room-type color parity |
| **Density** | Realistic room count, type, and geometry distributions |
| **Coverage** | Multiple samples per outline from stochastic flow sampling |
| **Validity** | Outside/overlap/gap = 0.0 via repair + reject policy |
| **Code quality (30%)** | Small modules, full test suite, documented |
| **Architecture (25%)** | Clean separation: data / repr / model / eval / demo |
| **Challenge alignment (25%)** | Vector output, flow model, outline-only conditioning |
| **Innovation (20%)** | MRR tokens, oracle gate, metric-aligned local evaluator |

---

## Slide 7: The Full System

| Component | Status | Location |
|---|---|---|
| `generate(outline)` contract | Done | `src/floorgen/generate.py` |
| Data preprocessing (MSD CSV) | Done | `src/floorgen/data/` |
| MRR representation + repair | Done | `src/floorgen/repr/` |
| Flow model + training | Done | `src/floorgen/model/` |
| Evaluation (FID, PRDC, validity) | Done | `src/floorgen/eval/` |
| Checkpoint loader + sampler | Done | `src/floorgen/model/sampler.py` |
| Post-train pipeline | Done | `src/floorgen/posttrain.py` |
| Gradio demo (local / Space-ready) | Done | `app.py`; no live URL committed |
| Batch export (Parquet/CSV) | Done | `scripts/export_batch.py` |

**Automated tests. Linter clean. One-command smoke test.**

---

## Slide 8: Verification Evidence

```
Smoke test results:
  generate(box(0,0,10,8))  -> 9 rooms
  generate(L-shaped)       -> 8 rooms

Validity metrics:
  outside_frac: 0.0000
  overlap_frac: 0.0000
  gap_frac:     0.0000
  invalid_rate: 0.0000

pytest:  55 tests passed, 1 skipped
ruff:    All checks passed
```

Verified smoke layouts are valid partitions of the input outline.
The evaluator reports outside, overlap, gap, and invalid-rate metrics.

---

## Slide 9: Local Demo

1. Pick a **real MSD apartment outline** (5 presets from actual Swiss buildings)
2. Generate **3-6 diverse layouts** from the same outline
3. Inspect **vector room polygons** in GeoJSON
4. Change seed for different valid arrangements
5. Backend auto-switches between baseline and trained checkpoint

> The UI shows provenance: which backend, what checkpoint, what settings.

**Same `generate(outline)` API the evaluator calls.**

---

## Slide 10: Honest Status

| Area | Status |
|---|---|
| Full pipeline (data, repr, model, eval, demo, export) | Implemented |
| Flow model architecture + training script | Implemented |
| Checkpoint loading via `FLOORGEN_CHECKPOINT` | Implemented |
| Baseline geometry validity | Perfect (0.0 outside/overlap/gap) |
| Trained checkpoint quality | In progress (early training) |
| Test-split generation export | Ready (pending final checkpoint) |

**The framework works end-to-end.** Final FID/density/coverage scores depend on the converged checkpoint being registered.

---

## Slide 11: Why We Win

1. **Research-grounded**: MRR choice comes directly from the MSD paper's own finding
2. **Evaluator-aligned**: local FID + PRDC + validity checks match organiser stack
3. **Clean engineering**: small modules, tested, documented, reproducible seeds
4. **Honest**: baseline fallback is never claimed as the scored model
5. **Complete**: data pipeline, model, evaluation, demo, export — all connected

> The system is built around what the judges actually measure, not what looks flashy.

---

## Appendix: Key References

- **MSD Paper**: van Engelenburg et al., "MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes", ECCV 2024. arXiv:2407.10121
- **HouseDiffusion**: Shabani et al., CVPR 2023 — diffusion over room corner coordinates
- **PRDC**: Naeem et al. (clovaai) — density and coverage metrics
- **Flow Matching**: Lipman et al., 2023 — conditional flow matching for generation

---

## Appendix: Repo Quick Start

```bash
# Install
uv sync --extra dev

# Smoke test (no data needed)
uv run --extra train python scripts/smoke_test.py

# Generate with default AMD checkpoint
python -c \
  "from shapely.geometry import box; from floorgen.generate import generate; \
   print(len(generate(box(0,0,10,8))), 'rooms')"

# Launch demo
python app.py
```

GitHub: https://github.com/jakobtfr/ehl-paris
