"""CLI script: evaluate generated layouts against reference outlines.

Runs the full pipeline: generate → validate → render → aggregate metrics.
Outputs a JSON report suitable for judges and CI dashboards.

Usage:
    python scripts/evaluate.py --demo
    python scripts/evaluate.py --outlines data/processed/outlines.parquet --n-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from floorgen.config import SEED
from floorgen.eval.metrics import distribution_metrics, validity_metrics
from floorgen.eval.render import RenderConfig, render_layout
from floorgen.generate import sample_layouts


def _demo_outlines(n: int = 5) -> dict:
    rng = np.random.default_rng(SEED)
    outlines = {}
    for i in range(n):
        w, h = rng.uniform(8, 16), rng.uniform(7, 13)
        outlines[f"demo_{i:03d}"] = box(0, 0, w, h)
    return outlines


def _load_outlines(path: Path) -> dict:
    import geopandas as gpd

    gdf = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    outlines = {}
    for i, row in gdf.iterrows():
        uid = str(row.get("unit_id", i))
        outlines[uid] = row.geometry
    return outlines


def evaluate(
    outlines: dict,
    n_samples: int = 4,
    seed: int = SEED,
    render_size: int = 512,
) -> dict:
    """Run full evaluation pipeline, return structured report."""
    cfg = RenderConfig(size=render_size)
    all_validity = []
    all_layouts_tuples = []
    rendered_images = []
    failures = []

    t0 = time.time()

    for unit_id, outline in outlines.items():
        try:
            layouts = sample_layouts(outline, seed=seed, n_samples=n_samples)
        except Exception as exc:
            failures.append({"unit_id": unit_id, "error": str(exc)})
            continue

        for layout in layouts:
            rooms_tuples = [(r["polygon"], r["label_idx"]) for r in layout]
            vm = validity_metrics(rooms_tuples, outline)
            vm["unit_id"] = unit_id
            all_validity.append(vm)
            all_layouts_tuples.append(rooms_tuples)

            img = render_layout(rooms_tuples, outline, cfg=cfg)
            rendered_images.append(img)

    elapsed = time.time() - t0

    # Aggregate validity
    n_layouts = len(all_validity)
    if n_layouts > 0:
        agg_validity = {
            "outside_frac_mean": np.mean([v["outside_frac"] for v in all_validity]),
            "overlap_frac_mean": np.mean([v["overlap_frac"] for v in all_validity]),
            "gap_frac_mean": np.mean([v["gap_frac"] for v in all_validity]),
            "invalid_rate_mean": np.mean([v["invalid_rate"] for v in all_validity]),
            "n_rooms_mean": np.mean([v["n_rooms"] for v in all_validity]),
            "perfect_partitions": sum(
                1 for v in all_validity
                if v["outside_frac"] < 0.01 and v["overlap_frac"] < 0.01
                and v["gap_frac"] < 0.05
            ),
        }
    else:
        agg_validity = {}

    # Distribution metrics
    dist = distribution_metrics(all_layouts_tuples) if all_layouts_tuples else {}

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "n_outlines": len(outlines),
        "n_samples_per_outline": n_samples,
        "n_layouts_generated": n_layouts,
        "n_failures": len(failures),
        "elapsed_seconds": round(elapsed, 2),
        "validity": {k: round(float(v), 4) for k, v in agg_validity.items()},
        "distribution": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in dist.items()
        },
        "failures": failures[:10],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated floor layouts")
    parser.add_argument("--outlines", type=Path, help="Path to outline geometries")
    parser.add_argument("--demo", action="store_true", help="Use demo outlines")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None, help="JSON report output path")
    args = parser.parse_args()

    if not args.outlines and not args.demo:
        parser.error("Provide --outlines <path> or --demo")

    outlines = _demo_outlines() if args.demo else _load_outlines(args.outlines)

    print(f"Evaluating {len(outlines)} outlines x {args.n_samples} samples...")
    report = evaluate(outlines, n_samples=args.n_samples, seed=args.seed,
                      render_size=args.render_size)

    print(f"\nResults ({report['elapsed_seconds']}s):")
    print(f"  Layouts generated: {report['n_layouts_generated']}")
    print(f"  Failures: {report['n_failures']}")
    if report["validity"]:
        print("  Validity:")
        for k, v in report["validity"].items():
            print(f"    {k}: {v}")
    if report["distribution"]:
        print("  Distribution:")
        print(f"    room_count_mean: {report['distribution'].get('room_count_mean', 'N/A')}")
        print(f"    room_count_std: {report['distribution'].get('room_count_std', 'N/A')}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to: {args.output}")
    else:
        print(f"\n{json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    main()
