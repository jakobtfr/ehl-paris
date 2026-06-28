"""Post-training checkpoint evaluation and export pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from shapely import wkt
from shapely.geometry.base import BaseGeometry

from .config import PATHS, SEED
from .eval.scoring import ScoreSummary, score_batch
from .export import ExportConfig, OutlineInput, export_to_csv, export_to_parquet
from .model.sampler import load_generator

ExportFormat = Literal["parquet", "csv"]


@dataclass(frozen=True)
class PostTrainResult:
    """Structured result from evaluating/exporting one trained checkpoint."""

    checkpoint: str
    checkpoint_sha256: str
    units: str
    split: str
    n_outlines: int
    n_samples: int
    seed: int
    sampler_steps: int
    presence_threshold: float
    device: str
    export_path: str | None
    report_path: str | None
    markdown_path: str | None
    score: dict[str, Any] | None
    checkpoint_train: dict[str, Any]
    dry_run: bool = False


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve checkpoint aliases such as ``mlp`` into concrete paths."""

    from .generate import resolve_checkpoint_reference

    return Path(resolve_checkpoint_reference(str(path))).expanduser()


def checkpoint_sha256(path: str | Path) -> str:
    """Return a stable checkpoint content hash for provenance sidecars."""

    path = resolve_checkpoint_path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_metadata(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load checkpoint metadata without requiring callers to know torch details."""

    import torch

    path = resolve_checkpoint_path(path)
    checkpoint = torch.load(path, map_location=device)
    return {
        "config": checkpoint.get("config", {}),
        "label_names": list(checkpoint.get("label_names", [])),
        "train": checkpoint.get("train", {}),
    }


def load_outlines_from_units(
    units_path: Path,
    *,
    split: str = "val",
    limit: int = 0,
) -> dict[str, BaseGeometry]:
    """Load evaluation/export outlines from processed ``units.jsonl`` records."""

    if not units_path.exists():
        raise FileNotFoundError(f"processed units file not found: {units_path}")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    outlines: dict[str, BaseGeometry] = {}
    with units_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            outline = wkt.loads(record["outline_wkt"])
            if outline.is_empty or outline.area <= 0:
                continue
            unit_id = str(record.get("unit_id", line_no))
            outlines[unit_id] = outline
            if limit and len(outlines) >= limit:
                break

    if not outlines:
        raise ValueError(f"no usable outlines found for split={split!r} in {units_path}")
    return outlines


def load_outline_records_from_units(
    units_path: Path,
    *,
    split: str = "val",
    limit: int = 0,
) -> dict[str, OutlineInput]:
    """Load outlines plus preserved unit/plan/floor metadata for exports."""

    if not units_path.exists():
        raise FileNotFoundError(f"processed units file not found: {units_path}")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    outlines: dict[str, OutlineInput] = {}
    with units_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            outline = wkt.loads(record["outline_wkt"])
            if outline.is_empty or outline.area <= 0:
                continue
            unit_id = str(record.get("unit_id", line_no))
            outlines[unit_id] = {
                "geometry": outline,
                "unit_id": record.get("unit_id", unit_id),
                "plan_id": record.get("plan_id"),
                "floor_id": record.get("floor_id"),
                "split": record.get("split"),
                "official_split": record.get("official_split"),
            }
            if limit and len(outlines) >= limit:
                break

    if not outlines:
        raise ValueError(f"no usable outlines found for split={split!r} in {units_path}")
    return outlines


def register_checkpoint_generator(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    steps: int = 32,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Install a checkpoint-backed sampler as ``floorgen.generate.GENERATOR``."""

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    generator = load_generator(
        checkpoint_path,
        device=device,
        steps=steps,
        threshold=threshold,
    )

    import floorgen.generate as generate_module

    generate_module.GENERATOR = generator
    return load_checkpoint_metadata(checkpoint_path, device=device)


def _score_payload(summary: ScoreSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["validity_score"] = summary.validity_score
    payload["overall_score"] = summary.overall_score
    return payload


def run_post_training(
    *,
    checkpoint_path: Path,
    units_path: Path = PATHS.processed_dir / "units.jsonl",
    output_dir: Path = PATHS.reports_dir / "post_train",
    split: str = "val",
    limit: int = 0,
    n_samples: int = 4,
    seed: int = SEED,
    steps: int = 32,
    threshold: float = 0.5,
    device: str = "cpu",
    export_format: ExportFormat = "parquet",
    dry_run: bool = False,
) -> PostTrainResult:
    """Load a checkpoint, run validation scoring, export layouts, and write reports."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if export_format not in ("parquet", "csv"):
        raise ValueError("export_format must be 'parquet' or 'csv'")

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    units_path = Path(units_path)
    output_dir = Path(output_dir)
    outline_records = load_outline_records_from_units(units_path, split=split, limit=limit)
    outlines = {
        unit_id: record["geometry"]
        for unit_id, record in outline_records.items()
        if isinstance(record, dict)
    }
    metadata = register_checkpoint_generator(
        checkpoint_path,
        device=device,
        steps=steps,
        threshold=threshold,
    )
    digest = checkpoint_sha256(checkpoint_path)

    base = PostTrainResult(
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=digest,
        units=str(units_path),
        split=split,
        n_outlines=len(outlines),
        n_samples=n_samples,
        seed=seed,
        sampler_steps=steps,
        presence_threshold=threshold,
        device=device,
        export_path=None,
        report_path=None,
        markdown_path=None,
        score=None,
        checkpoint_train=dict(metadata.get("train", {})),
        dry_run=dry_run,
    )
    if dry_run:
        return base

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = score_batch(outlines, n_samples=n_samples, seed=seed)
    if summary.n_layouts == 0 or summary.room_count_mean <= 0.0:
        raise RuntimeError(
            "checkpoint generated no rooms; adjust training, sampler steps, or presence threshold"
        )
    export_cfg = ExportConfig(
        output_dir=output_dir / "layouts",
        n_samples=n_samples,
        seed=seed,
        checkpoint=str(checkpoint_path),
        config_notes=json.dumps(
            {
                "checkpoint_sha256": digest,
                "sampler_steps": steps,
                "presence_threshold": threshold,
                "device": device,
                "split": split,
            },
            sort_keys=True,
        ),
    )
    export_path = (
        export_to_parquet(outline_records, export_cfg)
        if export_format == "parquet"
        else export_to_csv(outline_records, export_cfg)
    )

    result = PostTrainResult(
        **{
            **asdict(base),
            "export_path": str(export_path),
            "report_path": str(output_dir / "post_train_report.json"),
            "markdown_path": str(output_dir / "post_train_summary.md"),
            "score": _score_payload(summary),
        }
    )
    report_path = Path(result.report_path or "")
    markdown_path = Path(result.markdown_path or "")
    report_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    markdown_path.write_text(summary.to_markdown() + "\n")
    return result
