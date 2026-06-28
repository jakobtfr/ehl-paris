"""Post-training checkpoint evaluation/export CLI.

Example:
    uv run --extra train python scripts/post_train.py \
      --checkpoint checkpoints/flow.pt \
      --units data/processed/units.jsonl \
      --output-dir reports/post_train \
      --split val --n-samples 8
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from floorgen.config import PATHS, SEED  # noqa: E402
from floorgen.posttrain import run_post_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and export generated layouts from a trained flow checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained .pt checkpoint.")
    parser.add_argument(
        "--units",
        type=Path,
        default=PATHS.processed_dir / "units.jsonl",
        help="Processed units.jsonl produced by floorgen-preprocess.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.reports_dir / "post_train",
        help="Directory for report, summary, and exported layouts.",
    )
    parser.add_argument("--split", default="val", help="Processed-data split to evaluate/export.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all matching outlines.")
    parser.add_argument("--n-samples", type=int, default=4, help="Candidates per outline.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=32, help="Euler sampler steps.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Presence threshold.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint and units file without scoring or exporting.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_post_training(
            checkpoint_path=args.checkpoint,
            units_path=args.units,
            output_dir=args.output_dir,
            split=args.split,
            limit=args.limit,
            n_samples=args.n_samples,
            seed=args.seed,
            steps=args.steps,
            threshold=args.threshold,
            device=args.device,
            export_format=args.format,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if result.dry_run:
        print("Post-training dry run OK")
        print(f"  Checkpoint: {result.checkpoint}")
        print(f"  Units: {result.units}")
        print(f"  Split/outlines: {result.split} / {result.n_outlines}")
        print(f"  Sampler: steps={result.sampler_steps}, threshold={result.presence_threshold}")
        return 0

    print("Post-training pipeline complete")
    print(f"  Checkpoint: {result.checkpoint}")
    print(f"  Outlines: {result.n_outlines} x {result.n_samples}")
    print(f"  Export: {result.export_path}")
    print(f"  Report: {result.report_path}")
    print(f"  Summary: {result.markdown_path}")
    if result.score:
        print(f"  Validity score: {result.score['validity_score']:.1f}")
        print(f"  Overall score: {result.score['overall_score']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
