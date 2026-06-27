# Data pipeline handoff (agent/data)

Owner: Agent 2 — Data Pipeline & Normalization.
Owned paths: `src/floorgen/data/*`, `tests/test_data*.py`, `docs/data-*.md`.

This file is the running handoff log. Newest iteration on top.

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
