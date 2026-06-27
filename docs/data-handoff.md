# Data pipeline handoff (agent/data)

Owner: Agent 2 — Data Pipeline & Normalization.
Owned paths: `src/floorgen/data/*`, `tests/test_data*.py`, `docs/data-*.md`.

This file is the running handoff log. Newest iteration on top.

---

## Real-data validation (verified against the actual MSD CSV)

The pipeline's assumptions were checked against the real `mds_V2_5.372k.csv`
(1,086,846 rows; streamed read-only, never extracted or committed). Confirmed facts:

- **Columns**: 17 total; all 7 required (`geom, unit_id, plan_id, floor_id, entity_type,
  entity_subtype, roomtype`) present. There are also `apartment_id`, `area_id`,
  `unit_usage`, etc. — per challenge.md, `unit_id` is the apartment key (confirmed).
- **entity_type**: `separator` 602,196 / `opening` 281,320 / `area` 203,330. The
  `area` filter is essential.
- **Label coverage**: all **17** distinct `entity_subtype` values are covered by
  `SUBTYPE_TO_ROOM` -> **0 unmapped** area rows. `roomtype` has 10 values, all matching
  `ROOM_NAMES` casing. The taxonomy in `config.py` is complete for this dataset.
- **Geometry**: all 203,330 area geoms are valid `POLYGON` WKT in metres; **0 empty**.
  (Multipolygons only arise from `build_outline`'s union, not from raw rows.)
- **Grouping / leakage**: 18,902 distinct `unit_id`; 5,372 `plan_id` == 5,372 `floor_id`
  (1:1 here). Every `floor_id` maps to exactly **one** `plan_id` (max plans/floor = 1), so
  the plan-grouped split is floor-leakage-safe.
- **Rooms per unit** (excluding blank unit_id): min 1, median 9, mean 9.31, p90 13,
  p95 15, **p99 18, max 37**. So `MAX_ROOMS_K=24` sits above p99 and below max — sensible;
  the report's `suggested_max_rooms_k` (~18) backs this up. 251 units have <2 rooms (skipped).
- **No-unit rows**: **27,405** area rows (13.5%) have blank/NaN `unit_id` and are dropped
  from grouping (now reported as `n_area_rows_no_unit`).
- **End-to-end on real geometry**: ran `build_outline -> fit_transform -> invert` on 6 real
  units. Each fuses to a single 80-93 m^2 Polygon; metric round-trip relative area error
  was at most **1.44e-16** (float-exact). The inverse transform is correct on real data.

Note: each real apartment carries ~3-4 `Structure` areas (SHAFT/ELEVATOR/VOID). They are
correctly labelled (never invented); repr/model owners should decide how to treat them.

---

## Iteration 8 — report area rows with no unit_id

**Files changed**
- `src/floorgen/data/preprocess.py` — `main()` computes `gdf["unit_id"].isna().sum()` and
  `write_outputs` reports it as `n_area_rows_no_unit`. Surfaces the 13.5% of real area rows
  dropped for missing unit_id (previously invisible).
- `tests/test_data_report.py` — end-to-end test extended with a blank-unit_id area row.

**Tests run**
- `uv run --extra dev pytest tests -q` -> 104 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new. MSD CSV is read locally (user-provided zip); not downloaded or
committed, per constraints.

---

## Iteration 7 — room-count distribution for MAX_ROOMS_K

**Files changed**
- `src/floorgen/data/preprocess.py` — new `room_count_distribution(records)` returning a
  rooms-per-unit `histogram`, `p50/p90/p95/p99`, and `suggested_max_rooms_k`
  (= ceil(p99), the smallest slot count fitting ~99% of units). Added to the report under
  `room_count_distribution`.
- `tests/test_data_roomdist.py` — new unit tests (uniform collapse, histogram, percentile
  monotonicity, suggested-slot bounds, empty-safe).
- `tests/test_data_report.py` — added an end-to-end assertion that the field reaches the
  report (note: JSON serialises histogram keys as strings).

**Why** — `config.MAX_ROOMS_K` is documented as "should be set from the processed
room-count distribution before training". The report now provides that distribution so the
slot count is a data-driven, auditable choice. I do **not** edit `config.py` (not my path);
this is a shared-file handoff candidate once real MSD numbers are in.

**Shared-file request (added)**
- `src/floorgen/config.py`: once preprocessing has run on real MSD data, set `MAX_ROOMS_K`
  from `room_count_distribution.suggested_max_rooms_k` in the report (owner of config.py).

**Tests run**
- `uv run --extra dev pytest tests -q` -> 103 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new.

---

## Iteration 6 — outline construction + shell selection tests

**Files changed**
- `tests/test_data_outline.py` — new, synthetic geometry only. No production change.

Covers the single-source-of-truth conditioning outline:
- `build_outline` empty input raises; single room round-trips; overlapping rooms fuse
  without double-counting area; edge-sharing rooms become one polygon; a wall gap within
  the bridge distance (0.2 m < 2 x 0.3 m) is fused; far-apart rooms stay a 2-part
  MultiPolygon.
- `largest_shell` passes polygons through, picks the biggest piece of a MultiPolygon,
  preserves holes in the chosen piece, and rejects non-polygonal input (TypeError).

Buffer-out/in rounds corners by ~1e-5, so `build_outline` area assertions use a ~1%
tolerance (topology is the point, not exact area).

**Tests run**
- `uv run --extra dev pytest tests -q` -> 96 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new.

**Coverage status** — `src/floorgen/data/` now has direct tests for every module:
outline (labels + geometry), normalize (transform round-trip), preprocess (load,
build_records, split, report, end-to-end main).

---

## Iteration 5 — end-to-end report completeness

**Files changed**
- `tests/test_data_report.py` — new, synthetic CSV only. No production change.

Drives `main()` over a synthetic CSV (mix of usable units, a single-room unit, a
malformed-WKT row, and a non-area row) and asserts the full output contract:
- `units.jsonl` (one line per usable unit) and `manifest.parquet` are written.
- report `n_units` / `n_skipped` / `n_invalid_geom_rows` counts are correct.
- `n_unmapped_rooms` and `unmapped_label_sources` flag the unknown label.
- `split_summary` is present with `plan_leakage == []`.
- `label_frequencies` covers the known room classes.

This is the first test that exercises `write_outputs` and the CLI `main()` path end to
end, so all report fields added in iterations 1, 2 and 4 are now covered.

**Tests run**
- `uv run --extra dev pytest tests -q` -> 83 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new.

---

## Iteration 4 — plan-level split summary + leakage check

**Files changed**
- `src/floorgen/data/preprocess.py` — new `split_summary(records, split)` returning
  `unit_counts`, `plan_counts`, `n_plans`, and `plan_leakage` (plan_ids that straddle
  train/val — always empty for a correct split, now auditable). Added to the report under
  a new `split_summary` key; existing `split_counts` kept for backward compatibility.
- `tests/test_data_split.py` — new. Determinism (seed + record-order independence),
  leakage-safety (every plan single split), plan-level val fraction, tiny-input guard,
  and `split_summary` counts/leakage detection. Minimal synthetic records only.

**Data assumptions (reaffirmed)**
- Split groups by `plan_id`; `val_frac` is applied at the plan level
  (`n_val = max(1, int(n_plans * val_frac))`). Reproducible from `SEED` alone, independent
  of record order (plan ids are sorted before the seeded shuffle).

**Tests run**
- `uv run --extra dev pytest tests -q` -> 78 passed.
- `uv run --extra dev ruff check src/floorgen/data tests` -> clean.

**Blockers** — none new.
**Next** — end-to-end `write_outputs`/`main` test on a synthetic CSV to lock full report
completeness (`n_unmapped_rooms`, `n_invalid_geom_rows`, `split_summary`).

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
