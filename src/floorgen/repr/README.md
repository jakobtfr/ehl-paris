# Representation and Repair

`floorgen.repr` owns the geometric contract between the learned generator and
the vector floor-plan output.

## MRR Token Contract

The primary representation is one minimum rotated rectangle token per room:

```text
(cx, cy, w, h, angle, label_idx)
```

- `cx`, `cy`: rectangle center in metres.
- `w`, `h`: rectangle side lengths. Tokens are canonicalized to `w >= h`.
- `angle`: rectangle orientation in radians, periodic modulo pi.
- `label_idx`: index into `floorgen.config.ROOM_NAMES`, clamped on decode.

`RoomMRR` normalizes direct construction and array decoding so model output,
hand-authored tests, and polygon encodings share the same convention. Invalid
or non-finite geometry decodes to an empty polygon and is skipped by repair.

## Deterministic Repair Contract

`repair_partition()` is a validity layer, not a partition generator. It may:

- clip generated rooms to the outline,
- resolve overlaps by deterministic claimed-area subtraction,
- fill only gaps whose total area is below `max_gap_frac`,
- accept only overlaps below `max_overlap_frac`,
- drop pieces below `min_area`,
- preserve valid polygon parts from `MultiPolygon` or `GeometryCollection`
  results.

It must not invent a full layout when the model output leaves large missing
regions or heavy overlaps. Those cases raise `RepairRejected` unless explicitly
called with `reject_large_repairs=False`.

## Inspection Helpers

- `geometry_iou(a, b)`: area IoU for Shapely geometries.
- `encode_decode_iou(poly)`: raw polygon-to-MRR-to-polygon fidelity.
- `partition_accounting(parts, outline)`: overlap, gap, and outside-outline
  fractions for repaired room parts.

## Focused Checks

```bash
uv run --extra dev pytest tests/test_repr.py tests/test_repair.py -q
uv run --extra dev ruff check src/floorgen/repr tests/test_repr.py tests/test_repair.py
```
