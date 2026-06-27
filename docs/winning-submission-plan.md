# Winning Submission Plan

## Thesis

Win by building the most evaluation-aligned system, not the fanciest model. The
challenge rewards a generated distribution of valid vector room partitions. The
highest-ROI strategy is a flow-matching generator over a vector layout
parameterisation that is valid by construction, plus deterministic repair and
metric-aware sample ranking.

The core idea: generate room tokens and wall-layout controls, then convert them
into clipped vector polygons with a rectilinear partition layer inside the
apartment outline. The generative model samples the room count, room labels,
room sites, target areas, and partition controls. Deterministic code may only
enforce vector validity: no gaps, no overlaps, no outside-outline area, and
clean polygons.

Use weighted Voronoi or power-diagram partitioning only if an oracle
reconstruction gate proves that it rasterizes like real floor plans. If it
creates too many diagonal or convex cells, switch to a Manhattan-constrained
partition, wall-line graph, or slicing-tree representation before training.

## Product the Judges See

- `generate(outline, seed=42, n_samples=1, mode="raw")` returns labelled room
  polygons in the original coordinate system.
- A compatibility wrapper exposes plain `generate(outline)` if the evaluator
  requires the exact one-argument signature.
- A live demo lets judges upload or select an outline, sample multiple plausible
  layouts, and inspect vector polygons.
- A pitch deck explains why the system is aligned with FID, density, coverage,
  vector-output constraints, and code-review criteria.
- The repo includes reproducible training, evaluation, validation plots, model
  weights, and a methodology writeup.

## Generator Contract

The official scored generator must sample from the trained diffusion or
flow-matching model. Heuristics are allowed only as deterministic validity
repair, evaluation baselines, or demo ablations unless organisers explicitly
approve otherwise.

Proposed API:

```python
def generate(outline, seed: int = 42, n_samples: int = 1, mode: str = "raw") -> list[dict]:
    ...
```

If organisers require a strict `generate(outline)` function, provide a wrapper
that calls this API with default seed `42`, `n_samples=1`, and `mode="raw"`,
then returns the first sample in the evaluator-expected shape.

Inputs:

- Shapely `Polygon` preferred.
- GeoJSON polygon accepted for demo/CLI convenience.
- Coordinates preserved in the original metric coordinate system.
- Holes and `MultiPolygon` inputs handled explicitly: either preserve valid
  holes, select the largest shell with a warning, or reject with a clear error
  depending on organiser guidance.

Outputs:

- list of samples, length `n_samples`
- each sample contains room records with:
  - `label`: expected challenge room-type string
  - `polygon`: Shapely `Polygon`
  - `geojson`: serializable polygon geometry
- polygons must partition the outline within tolerance
- labels must be from the documented taxonomy
- same outline plus same seed must produce stable output

Contract tests:

- no invalid polygons
- no room outside the outline beyond tolerance
- no pairwise overlap beyond tolerance
- union area matches outline area within tolerance
- all rooms have labels
- seed `42` is deterministic
- GeoJSON serialization round-trips

## Architecture

### 1. Data Pipeline

Build a reproducible MSD preprocessing pipeline:

- Load `mds_V2_5.372k.csv`.
- Filter `entity_type == "area"`.
- Repair invalid geometries with a logged, deterministic policy.
- Canonicalise room labels into a small stable taxonomy: living, kitchen,
  bedroom, bathroom, corridor, storage/utility, balcony/other if present.
- Construct the outline exactly as specified: buffer rooms by `0.3`, union, then
  buffer back by `-0.3`.
- Normalise each plan for modelling: translate to origin, scale by square root
  of outline area, store inverse transform for final output.
- Save train/validation splits by `plan_id`, with seed `42`.

Deliverables:

- `data/processed/*.parquet` or `.geojsonl`
- split manifest
- preprocessing report with plan counts, room counts, label frequencies,
  invalid geometry counts, and outline complexity stats

### 2. Vector Representation

Represent each layout as a variable-size set of room tokens:

```text
room_token = (x, y, log_area, aspect_hint, room_type, presence, wall_controls)
```

Condition on the outline using:

- resampled boundary points, for example 128 points along the exterior
- scalar descriptors: area, perimeter, compactness, bounding-box ratio, number
  of vertices
- optional raster preview only as a conditioning feature, never as the output

Convert generated tokens to final polygons with a deterministic vectorizer:

- infer rectilinear partition lines or cells from room sites, target areas, and
  wall controls
- optionally create Manhattan-constrained weighted cells if oracle
  reconstruction validates the look
- clip cells to the input outline
- assign each cell the generated room type
- merge or remove tiny cells by deterministic rules
- snap, simplify, and validate polygons
- ensure returned polygons partition the outline

Why this is strong:

- Vector polygons are produced directly.
- Gaps and overlaps are structurally avoided.
- Rectilinear cells better match architectural floor plans after rasterization.
- Room areas and labels remain generative.
- Diversity comes from sampling token sets.
- The repair layer makes outputs robust under hidden evaluation.

Oracle reconstruction gate:

- encode real room polygons into the proposed token representation
- run only the deterministic vectorizer
- rasterize reconstructed layouts beside the original real layouts
- compute reconstruction IoU, room-area error, boundary error, and local FID
- require visual/metric acceptability before model training
- if the gate fails, change the representation before spending time on neural
  training

### 3. Generative Model

Use conditional flow matching over room-token sets.

Model shape:

- outline encoder: PointNet or small Transformer over boundary points
- room-token denoiser/vector field: Set Transformer or Transformer decoder with
  room slots
- count head: predicts room-count distribution conditioned on outline
- type head: predicts categorical room labels
- continuous head: predicts token coordinates, area, and shape hints

Training:

- train from scratch
- seed everything with `42`
- map unordered ground-truth rooms to slots with Hungarian matching over room
  type, centroid, area, and aspect/shape features; keep a deterministic
  canonical fallback sorted by type, area, then centroid for debugging
- losses: flow-matching loss for continuous token variables, cross-entropy for
  room count and labels, auxiliary losses for area distribution and room-count
  calibration
- augment with rotations, mirrors, coordinate jitter, and small outline
  simplification noise
- keep augmentations invertible and label-preserving

Fallback if time is tight:

- train a smaller conditional diffusion/flow model over fixed `K` room slots
- use presence logits for variable room counts
- still keep the vectorizer and repair layer
- keep the heuristic sampler as a baseline and emergency demo only, not the
  official scored generator unless approved

### 4. Metric-Aligned Local Evaluation

Build a local evaluator before model tuning. Default settings:

- raster size: `256 x 256`
- padding: 5% of the outline bounding box
- feature extractor: ImageNet InceptionV3 pool features as the baseline, unless
  the official harness specifies a different encoder
- density/coverage `k`: `5`
- validation sample count: `5` samples per validation outline for diversity
  checks, plus `1` default sample for evaluator-compatibility checks

Evaluator outputs:

- rasterize real and generated layouts with one fixed style
- compute local FID using a stable feature extractor
- compute density and coverage in the same feature space as local FID:
  - fit the real-layout feature manifold with `k` nearest neighbours
  - density: average number of real-neighbour radii containing generated
    features, normalised by `k`
  - coverage: fraction of real features whose nearest generated feature falls
    within the real `k`-NN radius
  - report metrics for raw samples and, if used, ranked samples
- track geometry validity: overlap area, gap area, outside-outline area, invalid
  polygon rate, tiny-room rate
- track distribution fit: room count, room type frequencies, area ratios,
  adjacency patterns, corridor presence

Assumption to verify with Davis AI: exact hidden rasterization settings. If the
official harness is unavailable, make the local evaluator explicit and
consistent, then show it in the deck.

### 5. Sampling and Ranking

Generate several candidate layouts per outline. Keep two modes:

- `mode="raw"`: return direct model samples after validity repair only.
- `mode="ranked"`: sample multiple candidates and rank them with deterministic
  validity and distribution heuristics.

Ranking rules:

- hard reject invalid geometries
- prefer near-zero gap/overlap/outside area
- penalise extreme tiny rooms
- match plausible room-count range for outline area
- match target room-type distributions
- prefer layouts with realistic area ratios and corridor/living/kitchen
  relationships

Use `raw` as the default until organisers confirm that sample ranking is allowed
and compatible with hidden coverage scoring. If ranking is allowed, tune the
ranker to reject broken layouts without collapsing every outline to the same
safe template.

## Hidden-Evaluator Hardening

Stress-test these cases before the final submission:

- outlines with holes
- `MultiPolygon` or nearly disconnected inputs
- high vertex-count outlines
- very small apartments
- very large apartments with many rooms
- rare room labels
- rooms smaller than the rasterization resolution
- invalid polygon orientation
- generated sites outside the outline
- disconnected clipped cells
- outlines whose buffer operation creates slivers

## Execution Timeline

### First 2 Hours: Make the Target Concrete

- Ask Davis AI for the exact evaluation harness or rasterization settings.
- Confirm allowed post-processing and sample ranking.
- Parse the MSD CSV and produce a first outline/rooms visualisation.
- Freeze label taxonomy and split manifest.
- Decide the official `generate()` input/output schema and write contract tests.

### Hours 2-6: Geometry-First Baseline

- Implement preprocessing.
- Implement vectorizer from room tokens to clipped polygons.
- Build heuristic token baseline from real room-count/type/area distributions.
- Create `generate(outline)` with deterministic seed control.
- Build validity metrics and visual panels.

Milestone: valid vector partitions for arbitrary outlines before any neural
model is trained.

Go/no-go: if valid `generate()` does not exist by hour 6, stop model work and
finish a correct vector API, evaluator, and demo skeleton first.

### Hours 6-10: Oracle Reconstruction Gate

- Encode real layouts into room tokens and wall controls.
- Reconstruct with the deterministic vectorizer only.
- Compare original vs reconstructed layouts visually and with IoU/boundary/local
  FID checks.
- Decide whether to keep rectilinear cells, switch to slicing-tree/wall-line
  representation, or use Manhattan-constrained power cells.

Go/no-go: if oracle reconstruction looks unlike real architecture by hour 10,
freeze neural work and repair the representation.

### Hours 10-14: Train the Flow Model

- Train fixed-slot conditional flow matching model.
- Use validation plots every few epochs.
- Compare against heuristic baseline.
- Tune room-count, label, and area-ratio losses.
- Save the best checkpoint by local FID plus validity.

Milestone: model beats the heuristic baseline on local realism/diversity without
collapsing to one template.

Go/no-go: if the model does not beat the heuristic baseline by hour 14, keep the
trained flow model as the official generator but spend remaining time on
validity, speed, demo, and methodology rather than architecture churn.

### Hours 14-20: Robustness and Judge-Facing Polish

- Add candidate sampling and ranking.
- Stress-test irregular outlines, small units, high vertex counts, and rare room
  counts.
- Make demo fast: cache model, return sample in seconds, show multiple samples.
- Add README commands for preprocessing, training, evaluation, and demo.

Sunday morning freeze: after hour 18, no architecture changes except clear bug
fixes. Spend the remaining time on reproducibility, demo reliability, deck
clarity, and submission packaging.

### Final Hours: Submission Package

- Freeze seed `42`.
- Export model weights.
- Produce representative generated layouts and diversity grids.
- Include pitch visuals:
  - blinded real-vs-generated grid
  - same-outline diversity grid
  - validity table
  - metric table against heuristic baseline
  - one slide explaining why the deterministic layer enforces vector validity
    without replacing the generative model
- Write methodology.
- Build pitch deck.
- Run clean clone smoke test.
- Ensure session branch exists: `entire/checkpoints/v1`.

## Judge-Score Alignment

### FID

- Match hidden raster appearance as closely as possible.
- Avoid visual artifacts through geometry-valid vectorization.
- Prefer rectilinear architectural partitions over generic cell diagrams.
- Use area/type/adjoining-room priors to stay on-distribution.

### Density

- Rank out obviously implausible samples.
- Penalise invalid geometry and weird room ratios.
- Calibrate room counts and labels to training distribution.

### Coverage

- Preserve stochastic sampling.
- Report multiple samples per outline in the demo.
- Avoid over-ranking into one safe mode.
- Use type/count/seed diversity metrics.

### Code Quality: 30%

- Small, readable modules.
- Deterministic configs.
- One-command train/evaluate/generate paths.
- Clear tests for geometry validity and API contract.

### Architecture: 25%

- Separate data, model, vectorizer, evaluator, and demo.
- Keep the vectorizer deterministic and the generator generative.
- Make assumptions explicit.

### Challenge Alignment: 25%

- Output vector polygons, not masks.
- Use diffusion or flow matching.
- Make the trained model responsible for count, labels, sites, areas, and wall
  controls.
- Train from scratch.
- Seed all data/training/sampling/evaluation paths with `42`.

### Innovation: 20%

- Valid-by-construction architectural partition layer with an oracle
  reconstruction gate.
- Distribution-aware sample ranking.
- Diversity controls in the demo.
- Honest local metric harness.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Hidden evaluator differs from local metrics | Ask for harness; keep rasterization configurable; optimize geometry validity and distribution stats that transfer. |
| Model training underperforms in hackathon time | Ship the trained small flow/diffusion model as the official generator; use the heuristic sampler only as a baseline or demo ablation unless approved. |
| Generated layouts collapse | Track coverage, sample entropy, room-count diversity, and show multiple samples per outline. |
| Invalid polygons hurt score | Use partition-by-construction vectorizer and validation tests. |
| Room labels are noisy | Preserve expected challenge labels at output; collapse rare labels internally only with an explicit mapping back to the official taxonomy. |
| Post-processing rules questioned | Keep generator central, document deterministic geometry layer, provide unranked mode. |
| Representation reconstructs real plans poorly | Run oracle reconstruction before training and switch representation if needed. |

## Organizer Questions

Ask these before implementation choices harden:

1. Can you share the exact local evaluation harness or rasterization settings
   used before FID, density, and coverage?
2. What are the most common ways a generated layout gets penalized or rejected?
3. Is the score driven more by semantic room labels, geometric partition realism,
   or matching the outline perfectly?
4. Are deterministic post-processing, geometry repair, and sample ranking allowed
   after diffusion or flow sampling?
5. What hidden-test-set edge cases should we expect: irregular outlines, holes,
   tiny apartments, rare room types, or unusual room counts?

## Definition of Done

- `generate(outline)` returns valid labelled vector polygons.
- Official generator samples from the trained diffusion/flow model, not a purely
  heuristic solver.
- Oracle reconstruction proves the vector representation can resemble real
  floor plans before model training.
- Local evaluation report compares real, heuristic baseline, and flow model.
- Demo shows at least five different plausible samples for one outline.
- README documents reproducible commands and seed handling.
- Deck maps choices directly to FID, density, coverage, and review criteria.
- Clean clone can run a smoke generation path without manual fixes.
