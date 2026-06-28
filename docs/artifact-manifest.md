# Artifact Manifest

Date: 2026-06-28

Large runtime artifacts are intentionally ignored by git via `.gitignore`.
This manifest records the local files used for the final smoke verification,
their hashes, and how to regenerate them. No Git LFS or external release asset
URL is configured yet, so a clean clone must either download the Kaggle dataset
and rerun the commands below or receive these files through an out-of-band
artifact handoff.

## Local Artifact Hashes

| Artifact | Local path | Size bytes | SHA256 | Git status |
| --- | --- | ---: | --- | --- |
| Kaggle archive | `data/raw/archive.zip` | 4,996,692,802 | `d4e64a85afb3d1efde1298973bbc5acfe9f4d5667d6d42861cfd61cdae4f66ae` | ignored |
| Primary AMD checkpoint | `checkpoints/flow-transformer-amd-862d422.pt` | 369,429,793 | `d47d6e083e65e301c44fd1ecb40b8c1326316fff73b4b39712bfe54d53fb70ca` | ignored |
| Processed units | `data/processed/units.jsonl` | 210,800,595 | `2b20af5758fa3c2bd55baf29cd2f70dedfe2ac64b13df39a0ac21f6130a1cb48` | ignored |
| Processed manifest | `data/processed/manifest.parquet` | 829,778 | `d12c63aac8c5a9e210ee6a3e4f0d308f6f64802f2a6942cd71dd4d19c744b421` | ignored |
| Preprocess report | `reports/preprocess_report.json` | 1,764 | `ac5310c01f3b4aac5adde2d7cab04c684ac99d38b0aa8f6b479c2eab21589853` | ignored |
| Test metrics smoke | `reports/final_test_metrics_smoke.json` | 4,175 | `2254bd3b1669b13ac986c3f999acfe9eeb44f8626954f93b55583e35fb584511` | ignored |
| Limited test export CSV | `outputs/final_test_export/layouts_20260628_115248.csv` | 67,937 | `5fb0923f53b0a0b2cd0b74db7cc54b3e8f8b32f567c585ac5359a93fd1531224` | ignored |
| Limited test export metadata | `outputs/final_test_export/layouts_20260628_115248_meta.json` | 1,511 | `5eea864b2e3025918f038178ffe85b316758f06bb0429e0370d63cec2336e0bc` | ignored |

## Regeneration Commands

Preprocess the extracted Kaggle archive. This uses the single geometry CSV plus
the official `train/full_out` and `test/full_out` floor-id markers:

```bash
uv run python -m floorgen.data.preprocess \
  --kaggle-dir data/raw/archive \
  --out data/processed \
  --reports reports
```

Expected split counts in `reports/preprocess_report.json`:

```json
{"train": 13499, "test": 2734, "val": 2418}
```

Run the final test metrics smoke:

```bash
uv run --extra train python scripts/evaluate.py \
  --units data/processed/units.jsonl \
  --split test \
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --threshold 0.5 \
  --mode ranked \
  --n-samples 1 \
  --real-metrics \
  --output reports/final_test_metrics_smoke.json
```

Observed smoke metrics:

```json
{
  "fid": 257.3317565917969,
  "density": 0.0,
  "coverage": 0.0,
  "n_real": 3,
  "n_generated": 3,
  "prdc_k": 2,
  "semantic_repair_count": 12
}
```

Generate the limited official test export smoke:

```bash
uv run --extra train python scripts/export_batch.py \
  --units data/processed/units.jsonl \
  --split test \
  --limit 3 \
  --checkpoint checkpoints/flow-transformer-amd-862d422.pt \
  --threshold 0.5 \
  --mode ranked \
  --n-samples 1 \
  --format csv \
  --output-dir outputs/final_test_export
```

The full intended export removes `--limit 3`. It was not run in this slice
because CPU generation over all 2,734 official test units is materially more
expensive than the smoke check.

## Current Gaps

- No external URL or release asset is configured for the checkpoint, processed
  data, metrics JSON, or generated export.
- The full 2,734-unit test export is not present; only a three-unit official
  test smoke export exists locally.
- The final pitch deck exists as Markdown (`docs/pitch-deck.md`), not as a
  rendered PDF/PPTX.
