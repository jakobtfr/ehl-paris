# Data pipeline handoff (agent/data)

Owner: Agent 2 — Data Pipeline & Normalization.
Owned paths: `src/floorgen/data/*`, `tests/test_data*.py`, `docs/data-*.md`.

This file is the running handoff log. Newest iteration on top.

---

## Iteration 3 — transform round-trip invariants

**Files changed**
- `tests/test_data_transform.py` — new, synthetic geometry only. No production change.

Locks the critical metric round-trip the renderer depends on:
- `fit_transform` scale is exactly `MODEL_SPACE_SIZE / max(dx, dy)` (= `256 / max(dx, dy)`),
  using the larger bbox dimension.
- normalized geometry fits the 256-unit box and is centred on the origin.
- `invert(normalize(g)) == g` (area + per-coordinate).
- the divide-by-zero guard returns `scale == 1.0` only for a true zero-extent point
  (a line still has one nonzero delta).
- `scale_features` values for a known rectangle (area, bbox, aspect, perimeter, compactness).

Note: writing these surfaced that the scale guard triggers only when *both* deltas are
zero — a degenerate line still scales normally. Behaviour is correct; the test now
documents it.

**Tests run**
- `uv run --extra dev pytest tests -q` -> 70 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new.

---

## Iteration 2 — tolerate invalid geometries on load

**Files changed**
- `src/floorgen/data/preprocess.py` — new `parse_area_geometries(df) -> (gdf, n_invalid)`
  and `_try_load_wkt(value)`. `_load` now returns `(gdf, n_invalid)` and delegates schema
  validation + area filtering + per-row WKT parsing to `parse_area_geometries`. Malformed,
  non-string (NaN), or empty geometries are dropped and counted instead of raising.
  `write_outputs` gains `n_invalid_geom` and reports `n_invalid_geom_rows`.
- `tests/test_data_load.py` — new, synthetic CSV/DataFrame only.

**Data assumptions (new this iteration)**
- A WKT cell is valid only if it is a non-empty string that `shapely.wkt.loads` accepts and
  the result is non-empty. Everything else is counted in `n_invalid_geom_rows` and dropped.
- `_load` return arity changed from `gdf` to `(gdf, n_invalid)`. Only `main` (my file) calls it.

**Tests run**
- `uv run --extra dev pytest tests -q` -> 33 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new.

---

## Iteration 1 — auditable unknown-label handling

**Files changed**
- `src/floorgen/data/outline.py` — added `classify_room(subtype, roomtype) -> (label, matched)`.
  `canonical_room_name` is now a thin wrapper returning the label only. Behaviour of
  the label mapping itself is unchanged; only the match/fallback flag is new.
- `src/floorgen/data/preprocess.py` — `build_records` now returns
  `(records, skipped, unmapped)`. Rows whose subtype/roomtype are not recognised are
  counted in `unmapped` keyed by `subtype=<x>|roomtype=<y>`. `write_outputs` accepts the
  counter and adds `n_unmapped_rooms` and `unmapped_label_sources` (top 20) to
  `reports/preprocess_report.json`. `main` threads the counter through.
- `tests/test_data_labels.py` — new, synthetic geometry only.

**Data assumptions**
- One `unit_id` == one apartment == one training example.
- Split groups by `plan_id` (no floor leakage). Missing `plan_id`/`floor_id` -> `-1`.
- Unknown subtype/roomtype deliberately maps to `Structure` (keeps the partition
  complete) but is now reported, never silently invented.
- Normalization scale is `MODEL_SPACE_SIZE / max(delta_x, delta_y)` = `256 / max(dx, dy)`,
  centred on the outline centroid; `PlanTransform.invert` returns metric coordinates.
- `_load` requires columns: geom, unit_id, plan_id, floor_id, entity_type,
  entity_subtype, roomtype; filters `entity_type == "area"`.

**Tests run**
- `uv run --extra dev pytest tests -q` -> 25 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers**
- `uv` was not installed on this machine; installed via `pip install --user uv`
  (uv 0.11.25). Environment otherwise per `uv.lock`.

**Shared-file requests (I do not own these — please apply)**
1. `.gitignore`: add Python artefacts so `__pycache__/` and `*.pyc` stop showing as
   untracked. Suggested block:
   ```
   __pycache__/
   *.py[cod]
   .venv/
   *.egg-info/
   .pytest_cache/
   .ruff_cache/
   .mypy_cache/
   ```

**Notes for other agents**
- `build_records` return arity changed from 2 to 3. Only `preprocess.main`/`write_outputs`
  (my files) call it today; if eval/demo start importing it, expect `(records, skipped, unmapped)`.
- Per rules I never push `main`. I keep `agent/data` rebased on `origin/main` and verify
  it merges cleanly; integration into `main` is left to whoever owns that.
