# Pitch Plan

## One-Sentence Pitch

We turn a bare apartment outline into several valid, labelled room layouts by
learning room geometry as simple vector tokens, then applying a small repair
step only to keep the output inside the outline.

## Core Story

The challenge is not to find one correct floor plan. It is to produce a family
of plausible plans that look like real Swiss apartments when judged against the
MSD distribution. Our approach follows the main lesson from the MSD paper:
complex real floor plans are hard because rooms are vector shapes with labels,
scale, topology, and irregular boundaries. We therefore avoid a pixel-only
shortcut and model rooms directly as vector geometry.

The paper shows why this matters. MSD contains 5,372 annotated floor plans,
18.9K+ apartments, and 165.3K+ labelled areas. It also reports higher graph
diversity for MSD than RPLAN and LIFULL: graph entropy 8.02 for MSD, versus
4.56 for RPLAN and 7.79 for LIFULL. In other words, this is not a toy
single-home layout problem; it is a real-distribution generation problem.

The paper also gives us the representation insight we used. Its diffusion
baseline struggled with full arbitrary room polygons, because MSD rooms can have
many corners. The minimum rotated rectangle (MRR) version performed better than
the full-polygon version: MHD+WCA improved average MIoU from 17.9 with polygons
to 21.8 with MRR, and compatibility from 71.1 to 76.2. Without WCA, MRR also
improved compatibility from 80.3 to 87.1. We used that same idea: compress each
room into a rotated rectangle token that is small enough to learn, while keeping
the output vector-based.

## Plain-English Approach

1. **Represent each room simply.**
   A room is stored as center point, width, height, rotation angle, room type,
   and whether the slot is present. This is the MRR representation from the MSD
   paper, adapted to the hackathon's apartment-outline setting.

2. **Condition only on the apartment outline.**
   The model sees sampled boundary points plus scale features such as area and
   bounding-box shape. It does not receive the target rooms, a target room graph,
   or a hand-built partition.

3. **Use a flow model to sample room tokens.**
   The model predicts a path from noise to room geometry. Fixed room slots and
   presence flags let it support different room counts without changing the
   output format.

4. **Repair validity, not design.**
   After sampling, deterministic code clips rooms to the outline, removes
   overlaps, and fills small slivers. If too much repair would be needed, the
   sample is rejected instead of letting the repair code become a rule-based
   floor-plan solver.

5. **Evaluate the way the judges evaluate.**
   The repo includes FID, density, and coverage plumbing, plus geometry checks
   for outside area, overlap, gaps, invalid polygons, room counts, and label
   diversity.

## Slide Plan

### 1. Title: From Apartment Outline to Plausible Rooms

**Message:** Given only the boundary, we generate labelled vector room polygons.

**Show:** Input outline on the left, 3 generated alternatives on the right.

**Outcome to say:** Same outline, multiple valid layouts, same vector format the
MSD renderer can consume.

### 2. Why This Is Hard

**Message:** The task rewards distribution fit, not a single answer.

**Show:** MSD numbers: 5,372 floor plans, 18.9K+ apartments, 165.3K+ areas,
graph entropy 8.02.

**Outcome to say:** The model must cover a broad real-world layout distribution:
low FID for realism, high density for on-manifold samples, and high coverage for
diversity.

### 3. The Paper Insight We Used

**Message:** Learn compact vector room geometry instead of full arbitrary
polygons or pixels.

**Show:** Paper comparison: full polygon MHD+WCA average MIoU 17.9 vs MRR+WCA
21.8; compatibility 71.1 vs 76.2.

**Outcome to say:** MRR tokens reduce the denoising problem while preserving
rotation and vector output. That is why our room token is center, size, angle,
type, and presence.

### 4. Our Architecture

**Message:** Outline in, room tokens out, repair layer last.

**Show:** Pipeline diagram:

```text
outline polygon
  -> boundary + scale features
  -> fixed-slot flow model
  -> MRR room tokens
  -> validity repair
  -> labelled room polygons
```

**Outcome to say:** The model decides room count, placement, shape, and type.
The repair layer only enforces the hard geometry contract.

### 5. Why It Aligns With Judging

**Message:** Every major judging metric maps to a part of the system.

**Show:**

| Judging signal | Our system response |
| --- | --- |
| FID | MSD-style rasterization and room-label color parity |
| Density | realistic room count, type, and geometry distributions |
| Coverage | multiple samples per outline from the flow model |
| Validity | outside/overlap/gap metrics plus repair/reject policy |
| Code review | small modules, tests, README, env example, smoke script |

**Outcome to say:** We optimized for the hidden evaluator's actual failure
modes: outside rooms, overlaps, gaps, implausible labels, and collapsed variety.

### 6. What Is Implemented

**Message:** The repo is inspectable and runnable-looking end to end.

**Show:** Module map or short checklist.

**Outcome to say:**

- `generate(outline)` returns labelled vector room polygons.
- Data preprocessing preserves `unit_id`, `plan_id`, `floor_id`, split metadata,
  outline construction, normalization, and reports.
- MRR encode/decode and repair are isolated in `src/floorgen/repr/`.
- Flow model, sampler, losses, and checkpoint loader are in `src/floorgen/model/`.
- Evaluation and export are in `src/floorgen/eval/`, `src/floorgen/export.py`,
  and `src/floorgen/posttrain.py`.
- Demo is a Gradio app with preset MSD outlines and GeoJSON output.

### 7. Verification Evidence

**Message:** The contract is locally checked.

**Show:** Latest command results.

**Outcome to say:**

- Smoke test passed on a rectangle and an L-shaped outline.
- Rectangle smoke output had 9 rooms with outside, overlap, gap, and invalid
  geometry all at `0.0000`.
- Multi-sample smoke generated 3 layouts and serialized GeoJSON.
- `uv run pytest -q` passed with one skipped test.
- `uv run ruff check src tests scripts` passed.
- `entire/checkpoints/v1` exists locally and on origin.

### 8. Honest Status and Remaining Work

**Message:** The framework is ready; final score depends on wiring real trained
weights and exports.

**Show:** Done / pending table.

**Outcome to say:**

| Area | Status |
| --- | --- |
| Contract, repair, demo, tests, evaluator, export | Implemented |
| Flow training path and checkpoint loader | Implemented |
| Default generator | Baseline fallback unless `FLOORGEN_CHECKPOINT` is set |
| Real MSD-trained checkpoint | Pending |
| Real processed data and test-split output export | Pending |

This is important to say plainly: the baseline is useful for testing the full
pipeline, but the scored submission should use the trained flow checkpoint.

## Quantitative Outcomes to Lead With

Use these only if they are current at presentation time:

| Outcome | Current value or target |
| --- | --- |
| Dataset scale from paper | 5,372 floor plans, 18.9K+ apartments, 165.3K+ areas |
| MSD graph diversity from paper | entropy 8.02 vs 4.56 RPLAN and 7.79 LIFULL |
| Paper MRR evidence | MHD+WCA MIoU 21.8 vs 17.9 for full polygons |
| Local smoke validity | 0.0000 outside / overlap / gap / invalid on rectangle smoke |
| Test suite | pytest passed with one skipped test |
| Lint | Ruff passed |
| Final model metrics | Pending until checkpoint + processed units are available for `--real-metrics` |
| Final geometry health | Pending final checkpoint/export run; baseline smoke is not the scored result |
| Test-time compute | Pending final export config: candidates per outline, seed range, ranking policy |

## Demo Script

1. Select a real MSD preset outline.
2. Generate 3 to 6 samples with the same seed.
3. Point out that the outline is unchanged across all outputs.
4. Point out room labels and vector polygons in the GeoJSON panel.
5. Change the seed to show diversity.
6. If a trained checkpoint is loaded, name the checkpoint and candidate count.
   If not, state that the demo is running the baseline backend while the same
   UI and contract will load the trained checkpoint through `FLOORGEN_CHECKPOINT`.

## Judge-Friendly Wording

Use this language in the deck:

- "We generate vector rooms directly, not pixels."
- "The model proposes the layout; repair only enforces validity."
- "MRR is a deliberate choice from the MSD paper: it keeps rotated rooms while
  making the generation problem smaller."
- "We evaluate realism and diversity as a distribution, because the challenge
  has no single correct plan per outline."
- "We document seeds, candidate counts, checkpoint hash, and export metadata so
  the score is reproducible."

Avoid this language unless the final checkpoint and outputs exist:

- "Fully trained production model"
- "Final FID/density/coverage"
- "Submitted weights are included"
- "State of the art"

## Closing Line

The practical win is that the system is built around the evaluator's real
constraints: vector output, outline preservation, labelled rooms, diverse
sampling, and reproducible exports. The research choice is grounded in MSD's
own finding that compact rotated-rectangle geometry is a better learning target
for complex real floor plans than full arbitrary polygons.

## Source Notes

- MSD paper: https://arxiv.org/abs/2407.10121
- Local verification run: `uv run python scripts/smoke_test.py`,
  `uv run pytest -q`, and `uv run ruff check src tests scripts`
