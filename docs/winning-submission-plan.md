# Winning Submission Plan

## Thesis

Win by building the most evaluation-aligned system, not the fanciest model. The
challenge rewards a generated distribution of realistic, diverse vector room
layouts. The highest-ROI strategy is a flow-matching generator that emits room
**geometry directly** (axis-aligned boxes or ordered corner sequences), plus a
deterministic layer that only repairs the output into a clean partition, plus a
metric-aware local evaluator.

The core idea: the generative model samples the room count, room labels, and the
actual room geometry (box corners or corner sequences) conditioned on the
outline. Keep as much of the geometry generative as possible. Deterministic code
may only enforce vector validity: snap to the outline, resolve small
gaps/overlaps, clip outside-outline area, and clean polygons. The deterministic
layer must not be what decides room *shape* — the challenge forbids a
"deterministic solver or purely rule-based partitioner", so shape diversity has
to come from the model.

Default representation: direct geometry. The challenge brief explicitly suggests
"rectangles or corner sequences" as room representations, and real Swiss rooms
are rectilinear, so direct boxes/corner sequences rasterize like real plans and
keep shape generative. Lean on published vector floor-plan generators
(HouseDiffusion diffuses room corner coordinates; House-GAN++ and the Modified
Swiss Dwellings baseline are directly relevant) instead of inventing a
representation from scratch, but adapt them to condition on **outline only** (no
input room graph), which is the harder setting here.

Alternative representations (weighted Voronoi / power diagram, slicing tree,
wall-line graph) are allowed only if the oracle reconstruction gate proves they
rasterize like real floor plans *and* leave enough shape decisions to the model
to stay clear of the rule-based-partitioner line. Expect Voronoi/power cells to
fail the gate (convex, diagonal cells unlike real plans); do not start there.

## Product the Judges See

- Primary scored artifact: the generated layouts for the held-out MSD split, in
  the organiser-requested format. Do not assume the judges need to run training
  during evaluation.
- `generate(outline, seed=42, n_samples=1, mode="raw")` returns labelled room
  polygons in the original coordinate system.
- A compatibility wrapper exposes plain `generate(outline)` if the evaluator
  requires the exact one-argument signature.
- A batch CLI exports all generations for a manifest of test outlines with seed
  `42`, fixed sample count, and metadata tying each output to the model
  checkpoint/config used.
- A live demo lets judges upload or select an outline, sample multiple plausible
  layouts, and inspect vector polygons.
- A pitch deck explains why the system is aligned with FID, density, coverage,
  vector-output constraints, and code-review criteria. It must explain the
  model architecture, room parameterisation, conditioning features, repair
  boundary, and sampling protocol.
- The repo includes reproducible training/evaluation code, validation plots, the
  checkpoint/config provenance, and a methodology writeup. Include model weights
  if practical; if not, make the code and logs clear enough that reviewers can
  see exactly how the submitted weights/generations were obtained.

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
- with `n_samples > 1`, the samples must be **distinct but deterministic**: seed
  `42` fixes the RNG, and successive draws differ from one another, so coverage
  is preserved while the run stays reproducible (the official protocol fixes a
  sample count and seed `42`, so the scored path draws multiple samples)

Contract tests:

- no invalid polygons
- no room outside the outline beyond tolerance
- no pairwise overlap beyond tolerance
- union area matches outline area within tolerance
- all rooms have labels
- seed `42` is deterministic for a single sample
- repeated `generate(outline, seed=42, n_samples=N)` calls reproduce the same N
  samples, and those N samples are not near-duplicates of each other
- GeoJSON serialization round-trips

## Architecture

### 1. Data Pipeline

Build a reproducible MSD preprocessing pipeline:

- Load `mds_V2_5.372k.csv`.
- Do not download or commit the Kaggle data in this repo. Read it from an
  explicit local path or `MSD_CSV_PATH` so the machine that has the dataset can
  run preprocessing without changing code.
- Filter `entity_type == "area"`.
- Treat one `unit_id` as one apartment/dwelling training example. Keep
  `plan_id` and `floor_id` as metadata and split groups so validation/test
  examples do not leak from the same floor/building context into training. The
  challenge data-construction script uses `unit_id`; `plan_id` is broader than a
  single apartment.
- Repair invalid geometries with a logged, deterministic policy.
- Canonicalise room labels into a small stable taxonomy, anchored on the actual
  label values present in the CSV for `entity_type == "area"` rows (inspect the
  distinct subtypes first; do not invent categories). Start from MSD
  `constants.py` instead of creating a private taxonomy: map `entity_subtype`
  through `ROOM_MAPPING` to the official `ROOM_NAMES` order
  (`Bedroom`, `Livingroom`, `Kitchen`, `Dining`, `Corridor`, `Stairs`,
  `Storeroom`, `Bathroom`, `Balcony`, plus non-room structural/door/window
  classes). Internally collapse rare labels only after preserving an explicit
  output mapping back to MSD names.
- Construct the outline exactly as specified: buffer rooms by `0.3`, union, then
  buffer back by `-0.3`. Use a compatibility helper that supports both
  GeoPandas/Shapely variants (`GeoSeries.union_all()` where available,
  `unary_union` otherwise).
- Normalise each plan's *shape* for modelling: translate to origin, scale by
  square root of outline area, store inverse transform for final output.
- Crucially, this makes every plan unit-area, which erases absolute scale. Store
  the pre-normalisation absolute area and bounding-box width/height as separate
  conditioning scalars, because room count and type mix depend on absolute size
  (a 30 m² studio vs a 120 m² flat). Without this the count head has no scale
  signal.
- Save train/validation splits by `unit_id`, grouped by `plan_id`/`floor_id`
  where possible, with seed `42`.

Deliverables:

- `data/processed/*.parquet` or `.geojsonl`
- split manifest with `unit_id`, `plan_id`, `floor_id`, room count, area, and
  label histogram columns
- preprocessing report with unit counts, grouped plan/floor counts, room counts,
  label frequencies, invalid geometry counts, and outline complexity stats

### 1a. MSD Compatibility Contract

Mirror the official MSD repository closely enough that our vectors can be
rendered by the organiser path without semantic drift:

- Keep an adapter from our generated room records to the MSD graph shape used by
  `plot.py`: a `networkx.Graph` whose room nodes have `geometry` as exterior
  coordinate lists, `room_type` as the integer index into `ROOM_NAMES`, and
  `centroid` as the room centroid.
- Generate only interior room nodes for the challenge output. Do not emit MSD
  structural/door/window classes unless the organiser schema explicitly asks
  for them.
- If the renderer draws graph edges, add a deterministic optional adjacency
  adapter: infer `passage` edges for rooms with boundary distance below the MSD
  graph-extraction tolerance (`0.04`), and leave `door`/`entrance` absent unless
  the submitted schema includes doors. Keep this off for pure polygon rendering.
- Preserve original coordinate units in exported vectors. The model can train in
  normalised coordinates, but the MSD renderer and outline construction expect
  metric coordinates.
- Add a tiny renderer smoke test that converts one generated layout to the MSD
  graph adapter and calls `plot_floor` headlessly.

### 2. Vector Representation

The model emits room geometry directly. Two candidate parameterisations, in
preference order:

1. **Axis-aligned boxes (default).** Each room token is
   `(cx, cy, w, h, room_type, presence)`. Simple, rectilinear, trivially valid,
   and a good match for the majority of Swiss rooms. The deterministic layer
   only snaps edges, resolves small overlaps/gaps, and clips to the outline.
2. **Ordered corner sequences (stretch).** Each room is a fixed-length sequence
   of corner coordinates with a stop/presence flag, à la HouseDiffusion. Handles
   L-shaped and non-rectangular rooms; more expressive but harder to keep valid.

Both keep the room *shape* generative, which is what the rules require. Pick (1)
first; only move to (2) if the oracle gate shows boxes can't reconstruct real
plans well enough.

Condition on the outline using:

- resampled boundary points, for example 128 points along the exterior
- **absolute scale scalars: total area (m²) and bounding-box width/height**, fed
  separately so the count/type heads can use real scale (see normalisation note
  in the data pipeline — shape is unit-area normalised, so absolute area is
  otherwise lost)
- shape scalars: perimeter, compactness, bounding-box aspect ratio, vertex count
- optional low-res raster of the outline as a conditioning feature only, never as
  the output

Deterministic validity-repair layer (repair only, not shape generation):

- snap room edges to each other and to the outline within a small tolerance
- resolve small pairwise overlaps and gaps by clipping/snapping shared edges
- clip rooms to the input outline
- merge or drop sub-resolution slivers by a fixed rule, then re-fill the freed
  area to the adjacent room so the partition stays gap-free
- simplify and validate polygons; assign each polygon its generated room type
- ensure returned polygons partition the outline within tolerance

Why this is strong:

- Vector polygons are produced directly by the model.
- Room shapes, areas, types, and count are all generative — safe on the
  "no rule-based partitioner" rule.
- Rectilinear geometry matches real floor plans after rasterization.
- Diversity comes from sampling the model, not from a partition heuristic.
- The repair layer makes outputs robust under hidden evaluation without taking
  over shape generation.

Oracle reconstruction gate:

- encode real room polygons into the chosen representation (fit boxes / extract
  corner sequences from real rooms)
- run only the deterministic repair layer
- rasterize reconstructed layouts beside the original real layouts, **using the
  same rasteriser the evaluator path uses**
- compute reconstruction IoU, room-area error, boundary error, and local FID
- require visual/metric acceptability before model training
- if the gate fails for boxes, move to corner sequences; if both fail, only then
  consider Voronoi/slicing — but re-check the rule-alignment risk first

### 3. Generative Model

Use conditional flow matching. Given the weekend time budget, the **primary**
model is the simpler fixed-slot design, not the variable-cardinality set model.

Primary model (build this first):

- fixed `K` room slots (K = max plausible room count, e.g. 12), each slot a
  continuous geometry vector (box `cx, cy, w, h` or corner sequence)
- presence logit per slot handles variable room counts without a separate
  set-cardinality mechanism
- outline encoder: PointNet or small Transformer over boundary points, plus the
  absolute-scale and shape scalars
- denoiser/vector field: small Transformer over the K slots conditioned on the
  outline encoding
- type head: categorical room label per slot
- losses: flow-matching loss for slot geometry, cross-entropy for presence and
  labels, auxiliary loss for area distribution / room-count calibration
- map unordered ground-truth rooms to slots with Hungarian matching over type,
  centroid, area, and aspect; keep a deterministic canonical ordering (type,
  area, centroid) as a debugging fallback
- train from scratch; seed everything with `42`
- augment with rotations, mirrors, coordinate jitter, and small outline
  simplification noise; keep augmentations invertible and label-preserving

Stretch model (only if the primary is solid and time remains):

- conditional flow matching over a variable-size **set** of room tokens with a
  Set Transformer and an explicit count head, removing the fixed-`K` cap
- higher ceiling on diversity and large apartments, but many more moving parts;
  do not let it block a working submission

Compute note: training a generative model from scratch to good FID in a few
hours is only realistic with a GPU. State the available hardware up front. If
GPU is limited, keep `K` small, the model small, and start training early
(overlap with pipeline work) rather than waiting for the oracle gate.

Baseline only, never the scored generator: a heuristic token sampler from real
room-count/type/area distributions. Use it as an evaluation baseline and
emergency demo fallback only, unless organisers explicitly approve otherwise.

### 4. Metric-Aligned Local Evaluation

Build a local evaluator before model tuning. The organisers clarified the
evaluation stack, so stop treating the metric path as unknown:

- **Density and coverage:** use the calculations from
  `clovaai/generative-evaluation-prdc`, specifically `compute_prdc` semantics
  over the same image features used for the judged raster set. Report density
  and coverage; precision/recall can be logged as diagnostics.
- **FID:** use PyTorch/TorchMetrics native FID:
  `from torchmetrics.image.fid import FrechetInceptionDistance`.
- **Rendering:** mirror the MSD official repository's `plot.py` path and
  constants. Do not tune against a custom Matplotlib style if it diverges from
  the official MSD renderer.

Implementation rule: vendor a small compatibility adapter or pin a dependency
reference for PRDC/MSD rendering, but keep our evaluator's public API stable.
Feed both real and generated plans through the same renderer, image size, axis
limits, colors, line widths, antialiasing, and tensor conversion before metric
updates. The plan with the best local score is the one that wins under this
rendering contract, not under a prettier demo renderer.

MSD renderer implications:

- The official `plot_floor(G, ax, node_size=50, edge_size=3)` path renders a
  floor-plan access graph, not a GeoDataFrame. It builds Shapely polygons from
  each node's `geometry`, chooses `room_type` if present otherwise
  `zoning_type`, colors rooms with `CMAP_ROOMTYPE`/`CMAP_ZONING`, draws room
  fills with black edges at `lw=0`, then overlays black graph nodes, black
  door/passage edges, and red entrance edges.
- Room type can affect rendered color when using the official room-type color
  map, so keep type prediction and type-to-color mapping aligned with MSD
  constants instead of treating labels as cosmetic.
- If the organisers' wrapper around `plot.py` strips graph edges/nodes or uses a
  polygon-only helper, match that exact wrapper in one config switch; the default
  should still be "official MSD renderer parity".
- Save representative rendered PNGs beside vector outputs so metric regressions
  can be inspected visually without reopening the raw CSV.

Default settings:

- raster size: match the organiser/MSD renderer wrapper; use `256 x 256` only as
  a fallback when no explicit size is exposed
- padding/axis limits: match the MSD rendering script/wrapper; keep
  `axis("equal")` and `axis("off")` in parity mode
- feature extractor: whatever TorchMetrics' `FrechetInceptionDistance` uses for
  the configured feature layer; keep the same features for PRDC unless the
  organiser harness says otherwise
- density/coverage `k`: `5`, matching the PRDC example/default convention unless
  the organiser harness overrides it
- diversity check: `5` samples per validation outline, plus `1` default sample
  for evaluator-compatibility checks
- FID/density/coverage sample budget: these are noisy on small sets, so compute
  them over a few hundred+ generated rasters total, not just a handful per
  outline

Evaluator outputs:

- rasterize real and generated layouts with the official MSD rendering style
- run the same generated-layout-to-MSD-graph adapter for evaluation and demo
  screenshots, with graph nodes/edges controlled by config
- compute local FID with `torchmetrics.image.fid.FrechetInceptionDistance`
- compute density and coverage with PRDC `compute_prdc` logic in the same
  feature space as local FID:
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

Remaining assumption to verify with Davis AI: the exact wrapper around MSD
`plot.py` for converting vector layouts to metric images, including image
resolution and whether graph nodes/edges are drawn. The metric implementations
themselves are no longer open questions.

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

## Engineering and Logistics

Code quality is the single largest scored criterion (30%), and several
submission requirements are pure logistics that sink teams late on Sunday.
Decide these on day one.

Repo layout and tooling:

- Package manager and runtime: `uv` with a single `pyproject.toml`; pin all deps
  (shapely, geopandas, torch, torchmetrics, torch-fidelity if required by
  TorchMetrics FID, numpy, matplotlib, scikit-learn/PRDC) with a committed
  lockfile.
- Module separation, one concern each: `data/` (preprocessing + splits),
  `repr/` (encode/decode geometry + validity repair), `model/` (flow model +
  training), `eval/` (rasteriser + FID/density/coverage), `generate.py`
  (the `generate(outline)` entry point), `demo/` (live app).
- Quality gates: `ruff` (lint + format), `mypy` on core modules, `pytest` for
  the contract tests and a representation round-trip test. Wire a minimal CI
  workflow so the review sees green checks.
- Config: one YAML/dataclass config holding seed, rasterisation params, model
  size, and `K`; no magic numbers scattered in code.
- Determinism utility: a single `seed_everything(42)` covering python/numpy/torch
  RNG; note that full CUDA training determinism needs extra cudnn flags and may
  cost speed — inference determinism for `generate` is the must-have.

Submission deliverables (all required by the brief):

- **Generated held-out split**: export the requested generation file(s) with
  seed `42`, sample count, checkpoint id, config hash, and renderer/evaluator
  version in sidecar metadata.
- **Live demo URL**: pick the host now (Gradio on Hugging Face Spaces or
  Streamlit Community Cloud). Cache the model at startup; aim for seconds per
  sample; show several samples per outline. Build the deploy path early — Sunday
  hosting surprises are a known time sink.
- **Model provenance**: the presentation and repo must make clear what model was
  trained, how rooms were parameterised, what config produced the submitted
  generations, and where the resulting checkpoint/weights came from. Commit
  weights via Git LFS or attach as a release asset if practical; otherwise keep
  training scripts, config, logs, and checkpoint metadata inspectable.
- **`generate(outline)` entry point**: export exactly this symbol as canonical;
  keep the extra knobs (`seed`, `n_samples`, `mode`) on a separate function so
  evaluator introspection of `generate` can't trip.
- **Working-session branch**: ensure `entire/checkpoints/v1` exists and captures
  the session record with at least one prompt; treat this as a checklist item
  from the start, not a final-hour scramble.
- **Methodology writeup + pitch deck**: keep a running notes file so the deck
  and writeup are assembled, not written from scratch at the end.

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
- generated room boxes lying partly or wholly outside the outline
- disconnected clipped rooms
- outlines whose buffer operation creates slivers

## Execution Timeline

### First 2 Hours: Make the Target Concrete

- Ask Davis AI for only the remaining evaluator details: the exact MSD `plot.py`
  wrapper, raster image size, axis/padding policy, and whether graph nodes/edges
  are drawn. Metric implementations are known: PRDC for density/coverage and
  TorchMetrics FID.
- Confirm allowed post-processing and sample ranking.
- Parse the MSD CSV by `unit_id` and produce a first outline/rooms
  visualisation.
- Freeze label taxonomy and split manifest.
- Decide the official `generate()` input/output schema and write contract tests.

### Hours 2-6: Geometry-First Baseline

- Implement preprocessing (including absolute-scale conditioning scalars).
- Implement encode/decode for the box representation plus the validity-repair
  layer (snap, clip, resolve overlaps/gaps).
- Build heuristic token baseline from real room-count/type/area distributions.
- Create `generate(outline)` with deterministic seed control.
- Build validity metrics, the MSD-`plot.py`-matching rasteriser, TorchMetrics
  FID, PRDC density/coverage, and visual panels.

Milestone: valid vector partitions for arbitrary outlines before any neural
model is trained.

Go/no-go: if valid `generate()` does not exist by hour 6, stop model work and
finish a correct vector API, evaluator, and demo skeleton first.

### Hours 6-10: Oracle Reconstruction Gate

- Encode real layouts into the box representation (fit boxes to real rooms).
- Reconstruct with the deterministic repair layer only.
- Compare original vs reconstructed layouts visually and with IoU/boundary/local
  FID checks.
- Decide whether boxes suffice, or escalate to corner sequences; treat
  Voronoi/slicing as a last resort and re-check rule alignment first.
- Start a small fixed-slot model training run as soon as the representation
  passes, overlapping with remaining gate work.

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
- Export generated layouts for the held-out split with checkpoint/config
  metadata.
- Export model weights if feasible; always export enough provenance to show how
  the submitted generations were obtained.
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

- Match the official MSD renderer appearance as closely as possible.
- Avoid visual artifacts through direct geometry plus validity repair.
- Prefer rectilinear architectural rooms over generic cell diagrams.
- Use area/type/adjoining-room priors to stay on-distribution.

### Density

- Compute locally with PRDC semantics against the rendered image features.
- Rank out obviously implausible samples.
- Penalise invalid geometry and weird room ratios.
- Calibrate room counts and labels to training distribution.

### Coverage

- Compute locally with PRDC semantics against the rendered image features.
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

- Separate data, representation/repair layer, model, evaluator, and demo.
- Keep the repair layer deterministic and the geometry generative.
- Make assumptions explicit.

### Challenge Alignment: 25%

- Output vector polygons, not masks.
- Use diffusion or flow matching.
- Make the trained model responsible for room count, labels, and the actual room
  geometry (box corners / corner sequences); restrict deterministic code to
  validity repair so the partition is not rule-based.
- Train from scratch.
- Seed all data/training/sampling/evaluation paths with `42`.

### Innovation: 20%

- Direct vector geometry generation with a deterministic validity-repair layer,
  gated by an oracle reconstruction check before training.
- Distribution-aware sample ranking.
- Diversity controls in the demo.
- Honest local metric harness.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Hidden evaluator differs from local metrics | Use PRDC for density/coverage, TorchMetrics FID, and MSD `plot.py` renderer parity; keep the exact renderer wrapper configurable. |
| Model training underperforms in hackathon time | Ship the trained small flow/diffusion model as the official generator; use the heuristic sampler only as a baseline or demo ablation unless approved. |
| Generated layouts collapse | Track coverage, sample entropy, room-count diversity, and show multiple samples per outline. |
| Invalid polygons hurt score | Generate geometry directly, then apply the deterministic validity-repair layer and validation tests. |
| Room labels are noisy | Preserve expected challenge labels at output; collapse rare labels internally only with an explicit mapping back to the official taxonomy. |
| Post-processing rules questioned | Keep generator central, document deterministic geometry layer, provide unranked mode. |
| Deterministic layer seen as a rule-based partitioner (violates the rules, hurts alignment 25%) | Generate room geometry directly (boxes/corner sequences); restrict the deterministic layer to validity repair only; document the boundary; avoid Voronoi/power-diagram shape generation. |
| Representation reconstructs real plans poorly | Run oracle reconstruction before training; start with boxes, escalate to corner sequences, only then alternative partitions. |
| Renderer wrapper unknown still flips details | Mirror MSD `plot.py` as the default rasteriser; keep image size, axis policy, node/edge drawing, and type palette config-swappable. |
| Live demo / generated split / weight provenance / session-branch logistics slip late | Decide host, output format, checkpoint provenance format, and `entire/checkpoints/v1` on day one; build the deploy path early. |

## Organizer Questions

Ask these before implementation choices harden:

1. Can you share the exact wrapper around MSD `plot.py` used before
   FID/density/coverage? In particular: image resolution, axis/padding policy,
   antialiasing/DPI, room-type palette, whether graph nodes/edges are drawn, and
   how submitted vector outputs are converted into the graph/polygon object the
   renderer expects.
2. How many samples does the harness draw per outline, and how is seed `42`
   applied across that sample count — one seed for the whole run with distinct
   successive draws, or re-seeded per sample?
3. What exact return shape does `generate(outline)` need (list of dicts, GeoJSON
   FeatureCollection, GeoDataFrame, list of WKT+label)?
4. What are the most common ways a generated layout gets penalised or rejected?
5. Is the score driven more by semantic room labels, geometric partition
   realism, or matching the outline perfectly?
6. Are deterministic post-processing, geometry repair, and sample ranking
   allowed after diffusion or flow sampling? Where is the line between allowed
   "validity repair" and a disallowed "rule-based partitioner"?
7. What hidden-test-set edge cases should we expect: irregular outlines, holes,
   tiny apartments, rare room types, or unusual room counts?
8. What exactly should be submitted for scoring: generated test-split file only,
   a runnable `generate(outline)` entry point, model weights, or all of the
   above?

## Definition of Done

- `generate(outline)` returns valid labelled vector polygons.
- Official generator samples from the trained diffusion/flow model, not a purely
  heuristic solver.
- Oracle reconstruction proves the vector representation can resemble real
  floor plans before model training.
- Local evaluation report compares real, heuristic baseline, and flow model.
- Local metrics use TorchMetrics FID and PRDC density/coverage on MSD-rendered
  images.
- Batch export produces the generated split with seed/checkpoint/config
  metadata.
- Demo shows at least five different plausible samples for one outline.
- README documents reproducible commands and seed handling.
- Presentation explains the model, parameterisation, sampling, repair layer, and
  how the submitted weights/generations were obtained.
- Deck maps choices directly to FID, density, coverage, and review criteria.
- Clean clone can run a smoke generation path without manual fixes.
