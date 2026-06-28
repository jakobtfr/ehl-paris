# Agent 4 Handoff: Evaluation, Rendering, and Batch Export

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `src/floorgen/eval/__init__.py` | Modified | Re-exports all eval symbols for clean API |
| `src/floorgen/eval/metrics.py` | Modified | Lazy torch imports, buffer(0) fix for invalid geom |
| `src/floorgen/eval/prdc.py` | Modified | sklearn lazy import with numpy fallback |
| `src/floorgen/eval/render.py` | Modified | Added render_batch() and save_render() |
| `src/floorgen/eval/scoring.py` | **New** | ScoreSummary with validity/overall score (0-100) |
| `src/floorgen/export.py` | **New** | Batch export: Parquet/CSV with WKT, metadata sidecar |
| `scripts/evaluate.py` | **New** | CLI: full generate→validate→render→report pipeline |
| `scripts/export_batch.py` | **New** | CLI: batch export to Parquet/CSV |
| `tests/test_eval.py` | **New** | 19 tests: validity, distribution, render, PRDC, imports |
| `tests/test_export.py` | **New** | 12 tests: schema, WKT, Parquet, CSV, metadata |
| `tests/test_eval_integration.py` | **New** | 11 end-to-end tests: full pipeline |
| `tests/test_eval_scoring.py` | **New** | 7 tests: scoring summary module |

## Output Schema

### Export Parquet/CSV columns
```
unit_id, sample_idx, seed, label, label_idx, wkt, area_m2,
v_outside_frac, v_overlap_frac, v_gap_frac, v_invalid_rate, v_n_rooms
```

### Metadata sidecar (JSON)
```json
{
  "timestamp": "20260627_155000",
  "n_outlines": 5,
  "n_samples_per_outline": 4,
  "seed": 42,
  "checkpoint": "baseline",
  "total_rooms_exported": 126,
  "columns": ["unit_id", ...],
  "room_labels": ["Bedroom", "Livingroom", ...]
}
```

### Evaluation report (JSON)
```json
{
  "validity": { "outside_frac_mean": 0.0, "overlap_frac_mean": 0.0, ... },
  "distribution": { "room_count_mean": 12.6, "label_freq": {...} },
  "n_layouts_generated": 10, "n_failures": 0
}
```

## Tests Run

All **77 tests** pass:
- `tests/test_eval.py` — 19 (validity, distribution, render, PRDC, lazy imports)
- `tests/test_export.py` — 12 (schema, WKT, Parquet, CSV, metadata, determinism)
- `tests/test_eval_integration.py` — 11 (end-to-end pipeline)
- `tests/test_eval_scoring.py` — 7 (scoring summary)
- `tests/test_contract.py` — 8 (pre-existing)
- `tests/test_repr.py` — 15 (pre-existing + other agents)
- `tests/test_repair.py` — 5 (pre-existing)

Commands:
```bash
python -m pytest tests/ -q          # 77 passed
python -m ruff check src/floorgen/eval tests scripts/evaluate.py scripts/export_batch.py
```

## Organizer-Renderer Assumptions

- Render size: 512×512 px (MSD default from `plot.py`)
- DPI: 100, no axes, equal aspect ratio
- Colour palette: centralized MSD-style 10-room mapping in `RenderConfig`
- Background: white (255, 255, 255)
- Edge: 1px black borders between rooms
- All configurable via `RenderConfig` dataclass; exact organiser wrapper details
  such as graph overlays remain an open parity question

## Blockers

Current blockers are packaging and final-scale artifacts, not the smoke path.
FID/PRDC computation requires `torch` + `torchmetrics` but now runs on the
3-unit official test smoke; full-scale metrics/export still need the ignored
local artifacts or an external artifact handoff.

## Shared-File Requests

1. **`.gitignore`** — already updated by agent/docs-demo (added `__pycache__/`, `*.pyc`)
2. **`pyproject.toml`** — would benefit from adding `geopandas` to dev deps for export tests, but not blocking since it's already in main deps

## Demo Commands

```bash
# Quick evaluation on demo outlines
python scripts/evaluate.py --demo --n-samples 4

# Batch export to Parquet
python scripts/export_batch.py --demo --format parquet

# Run all eval tests
python -m pytest tests/test_eval*.py tests/test_export.py -q
```
