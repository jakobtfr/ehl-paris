"""Batch export of generated layouts.

Exports layouts as a structured dataset with:
  - unit_id: identifier for the source outline
  - room labels and label indices
  - WKT/geom-compatible polygon representations
  - generation seed and candidate count
  - checkpoint/config metadata sidecar

Output format is Parquet (via pandas/pyarrow) for interoperability with
GIS tooling, the organiser's evaluation scripts, and downstream analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry.base import BaseGeometry

from .config import PATHS, ROOM_NAMES, SEED
from .eval.metrics import validity_metrics
from .generate import sample_layouts

OutlineInput = BaseGeometry | dict[str, Any]


@dataclass
class ExportConfig:
    """Controls batch export behaviour."""

    output_dir: Path = field(default_factory=lambda: PATHS.reports_dir / "export")
    n_samples: int = 4
    seed: int = SEED
    checkpoint: str = "baseline"
    mode: str = "raw"
    candidate_budget: int | None = None
    config_notes: str = ""
    include_validity: bool = True
    fail_on_error: bool = True


def _room_to_record(
    room: dict,
    unit_id: str,
    sample_idx: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten a single room dict into an export row."""
    poly = room["polygon"]
    record = {
        "unit_id": unit_id,
        "sample_idx": sample_idx,
        "seed": seed,
        "label": room["label"],
        "label_idx": room["label_idx"],
        "geom": poly.wkt,
        "wkt": poly.wkt,
        "area_m2": poly.area,
    }
    if metadata:
        for key in ("plan_id", "floor_id", "split", "official_split"):
            if key in metadata:
                record[key] = metadata[key]
    return record


def _outline_and_metadata(entry: OutlineInput) -> tuple[BaseGeometry, dict[str, Any]]:
    """Accept bare geometries or processed-unit records with id metadata."""
    if isinstance(entry, BaseGeometry):
        return entry, {}
    outline = entry.get("geometry")
    if not isinstance(outline, BaseGeometry):
        raise TypeError("outline record must contain a Shapely geometry field")
    return outline, dict(entry)


def export_layouts(
    outlines: dict[str, OutlineInput],
    cfg: ExportConfig | None = None,
) -> pd.DataFrame:
    """Generate and export layouts for a batch of outlines.

    Parameters
    ----------
    outlines : dict mapping unit_id -> Shapely outline geometry, or to a
        processed-unit metadata dict containing a ``geometry`` field
    cfg : export configuration (defaults to ExportConfig())

    Returns
    -------
    DataFrame with one row per room, columns:
        unit_id, sample_idx, seed, label, label_idx, geom, wkt, area_m2,
        optional plan/floor/split metadata, and optionally validity metrics per
        sample. ``geom`` is the MSD-compatible WKT column; ``wkt`` is retained
        for older local tooling.
    """
    if cfg is None:
        cfg = ExportConfig()
    if cfg.n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

    for unit_id, entry in outlines.items():
        outline, metadata = _outline_and_metadata(entry)
        try:
            layouts = sample_layouts(
                outline,
                seed=cfg.seed,
                n_samples=cfg.n_samples,
                mode=cfg.mode,
                candidate_budget=cfg.candidate_budget,
            )
        except Exception as exc:
            failures.append((unit_id, str(exc)))
            if cfg.fail_on_error:
                raise RuntimeError(f"layout generation failed for unit_id={unit_id}: {exc}") from exc
            continue

        for sample_idx, layout in enumerate(layouts):
            for room in layout:
                row = _room_to_record(room, unit_id, sample_idx, cfg.seed, metadata)
                rows.append(row)

            if cfg.include_validity:
                rooms_tuples = [(r["polygon"], r["label_idx"]) for r in layout]
                vm = validity_metrics(rooms_tuples, outline)
                for row in rows[-len(layout):]:
                    row.update({f"v_{k}": v for k, v in vm.items()})

    if not rows:
        detail = f"; first failure for {failures[0][0]}: {failures[0][1]}" if failures else ""
        raise RuntimeError(f"layout export produced no room rows{detail}")

    df = pd.DataFrame(rows)
    df.attrs["failures"] = [{"unit_id": unit_id, "error": error} for unit_id, error in failures]
    return df


def export_to_parquet(
    outlines: dict[str, OutlineInput],
    cfg: ExportConfig | None = None,
) -> Path:
    """Full pipeline: generate, validate, and write Parquet + metadata sidecar."""
    if cfg is None:
        cfg = ExportConfig()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = export_layouts(outlines, cfg)

    ts = time.strftime("%Y%m%d_%H%M%S")
    parquet_path = cfg.output_dir / f"layouts_{ts}.parquet"
    df.to_parquet(parquet_path, index=False)

    metadata = {
        "timestamp": ts,
        "n_outlines": len(outlines),
        "n_samples_per_outline": cfg.n_samples,
        "seed": cfg.seed,
        "checkpoint": cfg.checkpoint,
        "mode": cfg.mode,
        "candidate_budget": cfg.candidate_budget,
        "config_notes": cfg.config_notes,
        "total_rooms_exported": len(df),
        "columns": list(df.columns),
        "room_labels": list(ROOM_NAMES),
        "failures": df.attrs.get("failures", []),
    }
    meta_path = cfg.output_dir / f"layouts_{ts}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return parquet_path


def export_to_csv(
    outlines: dict[str, OutlineInput],
    cfg: ExportConfig | None = None,
) -> Path:
    """Like export_to_parquet but writes CSV for simpler toolchains."""
    if cfg is None:
        cfg = ExportConfig()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df = export_layouts(outlines, cfg)

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = cfg.output_dir / f"layouts_{ts}.csv"
    df.to_csv(csv_path, index=False)
    metadata = {
        "timestamp": ts,
        "n_outlines": len(outlines),
        "n_samples_per_outline": cfg.n_samples,
        "seed": cfg.seed,
        "checkpoint": cfg.checkpoint,
        "mode": cfg.mode,
        "candidate_budget": cfg.candidate_budget,
        "config_notes": cfg.config_notes,
        "total_rooms_exported": len(df),
        "columns": list(df.columns),
        "room_labels": list(ROOM_NAMES),
        "failures": df.attrs.get("failures", []),
    }
    meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    return csv_path
