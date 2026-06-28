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
from shapely import wkt
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from floorgen.config import SEED
from floorgen.eval.metrics import distribution_metrics, validity_metrics
from floorgen.eval.realism import try_image_distribution_report
from floorgen.eval.render import RenderConfig, render_layout
from floorgen.generate import backend_provenance, sample_layouts


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


def _load_units(
    path: Path,
    *,
    split: str | None = None,
    limit: int = 0,
) -> tuple[dict, dict]:
    outlines = {}
    real_layouts = {}
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            unit_id = str(record.get("unit_id", line_no))
            outline = wkt.loads(record["outline_wkt"])
            rooms = [
                (wkt.loads(room["wkt"]), int(room["label_idx"]))
                for room in record.get("rooms", [])
            ]
            if outline.is_empty or not rooms:
                continue
            outlines[unit_id] = outline
            real_layouts[unit_id] = rooms
            if limit and len(outlines) >= limit:
                break
    if not outlines:
        detail = f" for split={split!r}" if split else ""
        raise ValueError(f"no usable processed units found{detail} in {path}")
    return outlines, real_layouts


def evaluate(
    outlines: dict,
    real_layouts: dict | None = None,
    n_samples: int = 4,
    seed: int = SEED,
    render_size: int = 512,
    checkpoint: str = "baseline",
    checkpoint_sha: str | None = None,
    steps: int | str | None = None,
    threshold: float | str | None = None,
    device: str | None = None,
    real_metrics: bool = False,
    prdc_k: int = 5,
) -> dict:
    """Run full evaluation pipeline, return structured report."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    cfg = RenderConfig(size=render_size)
    all_validity = []
    all_layouts_tuples = []
    rendered_images = []
    real_images = []
    failures = []

    t0 = time.time()

    for unit_id, outline in outlines.items():
        if real_metrics and real_layouts and unit_id in real_layouts:
            real_images.append(render_layout(real_layouts[unit_id], outline, cfg=cfg))
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
    image_metrics = None
    if real_metrics:
        if real_images and rendered_images:
            image_metrics = try_image_distribution_report(
                np.stack(real_images, axis=0),
                np.stack(rendered_images, axis=0),
                prdc_k=prdc_k,
            )
        else:
            image_metrics = {
                "status": "blocked",
                "error": "no paired real/generated rendered images available",
                "n_real": len(real_images),
                "n_generated": len(rendered_images),
                "prdc_k": prdc_k,
                "fid": None,
                "precision": None,
                "recall": None,
                "density": None,
                "coverage": None,
            }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "n_outlines": len(outlines),
        "n_samples_per_outline": n_samples,
        "n_layouts_generated": n_layouts,
        "n_failures": len(failures),
        "elapsed_seconds": round(elapsed, 2),
        "backend": {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "sampler_steps": steps,
            "presence_threshold": threshold,
            "device": device,
        },
        "renderer": {
            "size": cfg.size,
            "pad_frac": cfg.pad_frac,
            "edge_width": cfg.edge_width,
            "background": cfg.bg,
        },
        "validity": {k: round(float(v), 4) for k, v in agg_validity.items()},
        "distribution": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in dist.items()
        },
        "image_metrics": image_metrics,
        "failures": failures[:10],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated floor layouts")
    parser.add_argument("--outlines", type=Path, help="Path to outline geometries")
    parser.add_argument("--units", type=Path, help="Processed units.jsonl with real room geometry")
    parser.add_argument("--split", default=None, help="Optional split filter for --units")
    parser.add_argument("--limit", type=int, default=0, help="Maximum units to load from --units")
    parser.add_argument("--demo", action="store_true", help="Use demo outlines")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None, help="JSON report output path")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional flow checkpoint to load")
    parser.add_argument("--steps", type=int, default=32, help="Euler sampler steps for checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Presence threshold for checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint inference")
    parser.add_argument(
        "--real-metrics",
        action="store_true",
        help="Compute FID and PRDC against real layouts from --units",
    )
    parser.add_argument("--prdc-k", type=int, default=5, help="k for PRDC density/coverage")
    args = parser.parse_args()

    if not args.outlines and not args.units and not args.demo:
        parser.error("Provide --outlines <path>, --units <units.jsonl>, or --demo")
    if args.real_metrics and not args.units:
        parser.error("--real-metrics requires --units")
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.prdc_k <= 0:
        parser.error("--prdc-k must be positive")

    real_layouts = None
    if args.units:
        outlines, real_layouts = _load_units(args.units, split=args.split, limit=args.limit)
    else:
        outlines = _demo_outlines() if args.demo else _load_outlines(args.outlines)
    provenance = backend_provenance()
    checkpoint = provenance["checkpoint"] or provenance["backend"] or "baseline"
    checkpoint_sha = None
    report_steps: str | int | None = provenance["steps"]
    report_threshold: str | float | None = provenance["presence_threshold"]
    report_device: str | None = provenance["device"]
    if args.checkpoint is not None:
        from floorgen.posttrain import checkpoint_sha256, register_checkpoint_generator

        register_checkpoint_generator(
            args.checkpoint,
            device=args.device,
            steps=args.steps,
            threshold=args.threshold,
        )
        checkpoint = str(args.checkpoint)
        checkpoint_sha = checkpoint_sha256(args.checkpoint)
        report_steps = args.steps
        report_threshold = args.threshold
        report_device = args.device

    print(f"Evaluating {len(outlines)} outlines x {args.n_samples} samples...")
    report = evaluate(
        outlines,
        real_layouts=real_layouts,
        n_samples=args.n_samples,
        seed=args.seed,
        render_size=args.render_size,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        steps=report_steps,
        threshold=report_threshold,
        device=report_device,
        real_metrics=args.real_metrics,
        prdc_k=args.prdc_k,
    )

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
    if report["image_metrics"]:
        print("  Image metrics:")
        if report["image_metrics"]["status"] == "ok":
            print(f"    fid: {report['image_metrics']['fid']}")
            print(f"    density: {report['image_metrics']['density']}")
            print(f"    coverage: {report['image_metrics']['coverage']}")
        else:
            print(f"    blocked: {report['image_metrics']['error']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to: {args.output}")
    else:
        print(f"\n{json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    main()
