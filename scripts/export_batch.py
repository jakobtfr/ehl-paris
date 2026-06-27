"""CLI script: batch-export generated layouts to Parquet/CSV.

Usage:
    uv run python scripts/export_batch.py --outlines data/processed/outlines.parquet
    uv run python scripts/export_batch.py --demo --format csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from floorgen.export import ExportConfig, export_to_csv, export_to_parquet


def _load_outlines(path: Path) -> dict:
    """Load outlines from a Parquet/GeoJSON file."""
    import geopandas as gpd

    gdf = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    outlines = {}
    id_col = "unit_id" if "unit_id" in gdf.columns else gdf.index.name or "idx"
    for i, row in gdf.iterrows():
        uid = row.get(id_col, str(i)) if id_col in gdf.columns else str(i)
        outlines[str(uid)] = row.geometry
    return outlines


def _demo_outlines(n: int = 3) -> dict:
    """Generate simple rectangular demo outlines."""
    rng = np.random.default_rng(42)
    outlines = {}
    for i in range(n):
        w, h = rng.uniform(8, 15), rng.uniform(8, 12)
        outlines[f"demo_{i:03d}"] = box(0, 0, w, h)
    return outlines


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-export generated floor layouts")
    parser.add_argument("--outlines", type=Path, help="Path to outline geometries file")
    parser.add_argument("--demo", action="store_true", help="Use demo rectangular outlines")
    parser.add_argument("--n-samples", type=int, default=4, help="Candidates per outline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", default="baseline")
    args = parser.parse_args()

    if not args.outlines and not args.demo:
        parser.error("Provide --outlines <path> or --demo")

    outlines = _demo_outlines() if args.demo else _load_outlines(args.outlines)

    cfg = ExportConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        checkpoint=args.checkpoint,
    )
    if args.output_dir:
        cfg = ExportConfig(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            checkpoint=args.checkpoint,
        )

    if args.format == "parquet":
        out = export_to_parquet(outlines, cfg)
    else:
        out = export_to_csv(outlines, cfg)

    print(f"Exported to: {out}")
    print(f"Outlines processed: {len(outlines)}")


if __name__ == "__main__":
    main()
