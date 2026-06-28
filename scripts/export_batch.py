"""CLI script: batch-export generated layouts to Parquet/CSV.

Usage:
    uv run python scripts/export_batch.py --outlines data/processed/outlines.parquet
    uv run python scripts/export_batch.py --demo --format csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from floorgen.export import ExportConfig, export_to_csv, export_to_parquet
from floorgen.generate import (
    DEFAULT_CANDIDATE_BUDGET,
    DEFAULT_DEVICE,
    DEFAULT_GENERATION_MODE,
    DEFAULT_PRESENCE_THRESHOLD,
    DEFAULT_SAMPLE_STEPS,
    backend_provenance,
    default_device,
)


def _checkpoint_notes_from_env() -> str:
    provenance = backend_provenance()
    notes: dict = {"backend": provenance}
    if provenance["checkpoint"]:
        steps = provenance["steps"]
        threshold = provenance["presence_threshold"]
        if steps is not None and int(steps) <= 0:
            raise ValueError("FLOORGEN_SAMPLE_STEPS must be positive")
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("FLOORGEN_PRESENCE_THRESHOLD must be between 0 and 1")

        from floorgen.posttrain import checkpoint_sha256

        notes["checkpoint_sha256"] = checkpoint_sha256(Path(provenance["checkpoint"]))
    return json.dumps(notes, sort_keys=True)


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
    parser.add_argument("--units", type=Path, help="Processed units.jsonl with outlines and metadata")
    parser.add_argument("--split", default=None, help="Optional split filter for --units")
    parser.add_argument("--limit", type=int, default=0, help="Maximum units to load from --units")
    parser.add_argument("--demo", action="store_true", help="Use demo rectangular outlines")
    parser.add_argument("--n-samples", type=int, default=4, help="Candidates per outline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional flow checkpoint to load")
    parser.add_argument(
        "--steps",
        type=int,
        default=int(DEFAULT_SAMPLE_STEPS),
        help="Euler sampler steps for checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(DEFAULT_PRESENCE_THRESHOLD),
        help="Presence threshold for checkpoint",
    )
    parser.add_argument(
        "--device",
        default=default_device() if DEFAULT_DEVICE == "auto" else DEFAULT_DEVICE,
        help="Torch device for checkpoint inference",
    )
    parser.add_argument("--mode", choices=["raw", "ranked"], default=DEFAULT_GENERATION_MODE)
    parser.add_argument("--candidate-budget", type=int, default=int(DEFAULT_CANDIDATE_BUDGET))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write successful rows even if some outlines fail generation",
    )
    args = parser.parse_args()

    if not args.outlines and not args.units and not args.demo:
        parser.error("Provide --outlines <path>, --units <units.jsonl>, or --demo")
    if sum(bool(value) for value in (args.outlines, args.units, args.demo)) > 1:
        parser.error("Provide only one of --outlines, --units, or --demo")
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.candidate_budget is not None and args.candidate_budget <= 0:
        parser.error("--candidate-budget must be positive")
    effective_device = default_device() if str(args.device).lower() == "auto" else args.device

    if args.units:
        from floorgen.posttrain import load_outline_records_from_units

        outlines = load_outline_records_from_units(
            args.units,
            split=args.split or "",
            limit=args.limit,
        )
    else:
        outlines = _demo_outlines() if args.demo else _load_outlines(args.outlines)
    provenance = backend_provenance()
    checkpoint = provenance["checkpoint"] or provenance["backend"] or "baseline"
    config_notes = _checkpoint_notes_from_env()
    if args.checkpoint is not None:
        from floorgen.posttrain import (
            checkpoint_sha256,
            register_checkpoint_generator,
            resolve_checkpoint_path,
        )

        resolved_checkpoint = resolve_checkpoint_path(args.checkpoint)
        metadata = register_checkpoint_generator(
            resolved_checkpoint,
            device=effective_device,
            steps=args.steps,
            threshold=args.threshold,
        )
        checkpoint = str(resolved_checkpoint)
        config_notes = json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha256(resolved_checkpoint),
                "sampler_steps": args.steps,
                "presence_threshold": args.threshold,
                "device": effective_device,
                "checkpoint_train": metadata.get("train", {}),
            },
            sort_keys=True,
        )

    cfg = ExportConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        checkpoint=checkpoint,
        mode=args.mode,
        candidate_budget=args.candidate_budget,
        config_notes=config_notes,
        fail_on_error=not args.allow_partial,
    )
    if args.output_dir:
        cfg = ExportConfig(
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            checkpoint=checkpoint,
            mode=args.mode,
            candidate_budget=args.candidate_budget,
            config_notes=config_notes,
            fail_on_error=not args.allow_partial,
        )

    if args.format == "parquet":
        out = export_to_parquet(outlines, cfg)
    else:
        out = export_to_csv(outlines, cfg)

    print(f"Exported to: {out}")
    print(f"Outlines processed: {len(outlines)}")
    print(f"Mode: {args.mode}; checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
