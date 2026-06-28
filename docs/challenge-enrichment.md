# Challenge Enrichment Notes

These notes capture unofficial, proprietary challenge intelligence gathered
during the Paris Hackathon. They are not part of the public challenge brief, but
they should inform modelling choices, validation checks, and local evaluation.

## Source Threads

Troy asked Amine Chraibi for clarification on the challenge track:

> Hey Amine,
>
> I'm currently at the Paris Hackathon and was hoping you could answer some
> questions I had for this track:
>
> Questions:
>
> "Can you share the exact local evaluation harness or rasterization settings
> used before FID/density/coverage?"
>
> "What hidden-test-set edge cases should we expect: irregular outlines, holes,
> tiny apartments, rare room types, or unusual room counts?"
>
> "Is the score driven more by semantic room labels, geometric partition
> realism, or matching the outline perfectly?"
>
> "What are the most common ways a generated layout gets penalized or
> rejected?"
>
> Thanks you so much

Amine replied:

> Hey Troy !
>
> - For density and coverage, we'll use the calculations from this repo:
>   https://github.com/clovaai/generative-evaluation-prdc. For FID we'll use
>   PyTorch's native FID. For rendering, we'll use the rendering scripts from
>   the MSD official repository:
>   https://github.com/caspervanengelenburg/msd
> - Don't worry about that, test set distribution is relatively similar to the
>   train set.
> - Outline must be perfectly kept, and all rooms must be contained inside the
>   outline. The score is driven by both semantic room labels and geometric
>   partition.
> - Rooms outside the outline, unrealistic geometric shapes, unrealistic room
>   disposition, e.g. bathroom inside kitchen.
>
> Good luck !

Discord clarification:

> aymen - 13:14
>
> Do we need to deliver the model or just the generation of the test split of
> the MSD dataset?

Hannes-Leonhard replied:

> hannes-leonhard - 13:22
>
> generally the generation is enough but your presentation should explain your
> model, parametrization and from your code we should see how your weights were
> obtained

Discord pitch-format clarification from 2026-06-27:

> flavia - 22:03
>
> hi we were wondering how long the pitch should be and what the structure
> should look like?

Hannes-Leonhard replied:

> hannes-leonhard - 22:34
>
> Around 5min, structure it like a research piece where you start with your
> method / architecture, and then show us your metrics (or the result of your
> auto research system)

Sunday roadmap deck pitch slide:

> how to pitch pt. 1
>
> the rule: kiss (keep it short and simple)
>
> show your methodology, architecture and results / metrics,
>
> 5min pitching, 5 min questions from the jury
>
> other teams listen to your pitch,
>
> no business slides
>
> Challenge 1 & 3; room valley
>
> Challenge 2; room palo

Troy asked Amine Chraibi a follow-up:

> Hey Amine, A few more questions (again):
>
> What would really impress the judges?
>
> What are the exact rendering settings used before FID/density/coverage?
> (image size, DPI, padding, colors, line widths, antialiasing, does plot_floor
> draw nodes/edges)
>
> What output format do you want for the generated test split? (GeoJSON, WKT +
> label CSV, polygons by unit_id, coordinate precision, required label names,
> metadata fields)
>
> How many samples per outline are evaluated and how is seed 42 applied? (global
> RNG, reset per outline, per sample, or fixed submitted file)
>
> What post-processing is allowed after the model samples geometry? (snapping,
> clipping to outline, overlap/gap resolution, sliver removal, sample ranking)
>
> Are room types used in metric rendering or only for code/presentation review?
> (colored by MSD room type, arbitrary per room, or binary occupancy)
>
> Thank you again

Amine replied:

> - be original !
> - MSD rendering script should have all settings (512x512 etc)
> - same format as the geom column in the dataset so rendering can be done with
>   the same script
> - seed isn't predetermined and use any test time compute method of your choice
>   (i.e generate as many samples as you want as long as you document it
>   properly)
> - everything is allowed as long as it's documented properly
> - room type will be used for rendering which impacts FID so yes

A later Discord clarification added three operational details:

- A rendered example using the MSD repository's palette/format was confirmed as
  the expected evaluation-style appearance.
- The room-type/color association is correct when it follows the official MSD
  color palette. Using a different palette from the initial snippet is fine if
  the colors come from the MSD repo mapping.
- Teams are expected to evaluate their own performance on FID, density, and
  coverage and report those numbers. There is no organiser-provided evaluation
  harness for this challenge.

A dataset-split clarification added:

- The Kaggle dataset has a predefined train/test split.
- Teams may split the provided training set into train/validation as they wish.
- The predefined train/test split is not already in the required outline format,
  so the conversion script should be applied without changing identity fields.
- `plan_id` values stay the same through conversion.

## Practical Interpretation

### Evaluation Stack

- Density and coverage should be locally approximated with the PRDC
  implementation:
  `https://github.com/clovaai/generative-evaluation-prdc`.
- FID should be computed with PyTorch's native FID implementation.
- Rasterization should follow the rendering scripts from the official MSD
  repository:
  `https://github.com/caspervanengelenburg/msd`.
- The MSD rendering script is the source of truth for settings such as `512x512`
  resolution, colors, line widths, padding, and related plot details.
- Room types affect rendering, which means semantic labels can affect FID
  directly through room colors.
- There is no organiser-provided evaluation harness. The submission should
  self-report FID, density, and coverage, including enough detail to reproduce
  the renderer, split, candidate budget, and metric settings.

Local validation should prioritize matching these dependencies and rendering
settings as closely as possible.

### Dataset Splits

- Respect the predefined Kaggle train/test split.
- Create any validation set only from the predefined training split.
- Preserve `plan_id` through outline-format conversion so split identity remains
  auditable.
- Do not create a fresh random train/test split from the full CSV.

### Submission Expectations

The generated outputs for the test split appear to be the primary required
delivery. However, the submission should still include enough supporting
material to make the model and training process auditable:

- The presentation should explain the model architecture.
- The presentation should explain the room/layout parametrization.
- The pitch should be around 5 minutes and structured like a research piece:
  start with method/architecture, then show metrics or the result of the
  automatic research/generation system.
- Plan for 5 minutes of pitching followed by 5 minutes of jury questions.
- Keep the deck short and simple, technical, and results-focused. Show
  methodology, architecture, and results/metrics; do not include business
  slides.
- Expect other teams to listen to the pitch.
- The code should make clear how the submitted weights were obtained.
- Training code and configuration should be reproducible enough for organisers
  to trace the path from dataset to weights to generated layouts.
- Generated test-split geometry should use the same format as the MSD `geom`
  column so the organisers can render it with the same script.
- Seed and sample count are not predetermined by the organisers. Any test-time
  compute strategy, including generating many candidates per outline, is allowed
  if documented clearly.
- Post-processing is allowed if documented clearly, including snapping,
  clipping, overlap/gap resolution, sliver removal, and sample ranking.

### Hidden Test Distribution

The hidden test set is expected to be close to the training distribution. We
should still validate robustness, but we do not need to over-optimize for exotic
outlines or rare adversarial cases unless they are present in the train set.

Key implication: spend more effort modelling the empirical MSD distribution than
inventing support for unlikely floor-plan shapes.

### Scoring Priorities

The outline is a hard constraint:

- Generated rooms must stay inside the apartment outline.
- The generated layout must preserve the input outline perfectly.
- Room polygons should form a plausible geometric partition.
- Semantic room labels matter alongside geometry.

This suggests a hierarchy for generation:

1. Satisfy outline containment and boundary preservation.
2. Produce realistic room geometry.
3. Assign plausible semantic labels.
4. Preserve train-set-like diversity for density and coverage.
5. Be original in the model/design story, not only valid.

### Likely Penalties and Rejections

Layouts are likely to be penalized for:

- Rooms outside the outline.
- Rooms that fail to respect the apartment boundary.
- Unrealistic geometric shapes.
- Unrealistic room dispositions.
- Semantically implausible adjacency or containment, such as a bathroom inside a
  kitchen.

## Implementation Implications

- Add geometric post-processing that clips or rejects any room polygon outside
  the outline.
- Add validation metrics for area outside outline, uncovered interior area,
  overlaps, invalid polygons, and disconnected rooms.
- Treat outline preservation as non-negotiable, even if it reduces generative
  freedom.
- Use train-set statistics for room counts, room labels, room areas, adjacency,
  and aspect ratios.
- Add semantic sanity checks for impossible or highly unlikely room
  relationships.
- Run local qualitative review using MSD-style rasterizations, not only vector
  plots.
- Package generated test-split layouts as the main deliverable, while keeping
  training code, configs, weights, and methodology clear enough to support the
  presentation.
- Include a self-evaluation report for FID, density, and coverage because the
  organisers are not providing an evaluation harness.
- Use a documented test-time compute pipeline: generate multiple candidates,
  repair/rank them if useful, and record the sample count, seeds, filters, and
  ranking criteria.
- Preserve and validate room labels because they affect rendered colors and
  therefore FID.

## Remaining Follow-Up Questions

- Exact PyTorch FID package or API version used by the organisers.
- Exact local renderer wrapper to use for our self-report, including whether
  graph nodes/edges should be drawn.
- Exact invalid-sample accounting convention for our self-report, even though
  post-processing itself is allowed when documented.
