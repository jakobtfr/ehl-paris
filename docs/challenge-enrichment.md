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

## Practical Interpretation

### Evaluation Stack

- Density and coverage should be locally approximated with the PRDC
  implementation:
  `https://github.com/clovaai/generative-evaluation-prdc`.
- FID should be computed with PyTorch's native FID implementation.
- Rasterization should follow the rendering scripts from the official MSD
  repository:
  `https://github.com/caspervanengelenburg/msd`.

Local validation should prioritize matching these dependencies and rendering
settings as closely as possible.

### Submission Expectations

The generated outputs for the test split appear to be the primary required
delivery. However, the submission should still include enough supporting
material to make the model and training process auditable:

- The presentation should explain the model architecture.
- The presentation should explain the room/layout parametrization.
- The code should make clear how the submitted weights were obtained.
- Training code and configuration should be reproducible enough for organisers
  to trace the path from dataset to weights to generated layouts.

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

## Open Follow-Up Questions

- Exact PyTorch FID package or API version used by the organisers.
- Exact MSD renderer command, image size, color mapping, antialiasing, and line
  width settings.
- Number of generated samples per outline during final scoring.
- Whether invalid samples are assigned worst score, filtered, or partially
  penalized.
