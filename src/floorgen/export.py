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


@dataclass
class ExportConfig:
    """Controls batch export behaviour."""

    output_dir: Path = field(default_factory=lambda: PATHS.reports_dir / "export")
    n_samples: int = 4
    seed: int = SEED
    checkpoint: str = "baseline"
    config_notes: str = ""
    include_validity: bool = True


def _room_to_record(
    room: dict,
    unit_id: str,
    sample_idx: int,
    seed: int,
) -> dict[str, Any]:
    """Flatten a single room dict into an export row."""
    poly = room["polygon"]
    return {
        "unit_id": unit_id,
        "sample_idx": sample_idx,
        "seed": seed,
        "label": room["label"],
        "label_idx": room["label_idx"],
        "wkt": poly.wkt,
        "area_m2": poly.area,
    }


def export_layouts(
    outlines: dict[str, BaseGeometry],
    cfg: ExportConfig | None = None,
) -> pd.DataFrame:
    """Generate and export layouts for a batch of outlines.

    Parameters
    ----------
    outlines : dict mapping unit_id -> Shapely outline geometry
    cfg : export configuration (defaults to ExportConfig())

    Returns
    -------
    DataFrame with one row per room, columns:
        unit_id, sample_idx, seed, label, label_idx, wkt, area_m2,
        and optionally validity metrics per sample.
    """
    if cfg is None:
        cfg = ExportConfig()

    rows: list[dict[str, Any]] = []

    for unit_id, outline in outlines.items():
        try:
            layouts = sample_layouts(
                outline, seed=cfg.seed, n_samples=cfg.n_samples, mode="raw"
            )
        except Exception:
            continue

        for sample_idx, layout in enumerate(layouts):
            for room in layout:
                row = _room_to_record(room, unit_id, sample_idx, cfg.seed)
                rows.append(row)

            if cfg.include_validity:
                rooms_tuples = [(r["polygon"], r["label_idx"]) for r in layout]
                vm = validity_metrics(rooms_tuples, outline)
                for row in rows[-len(layout):]:
                    row.update({f"v_{k}": v for k, v in vm.items()})

    df = pd.DataFrame(rows)
    return df


def export_to_parquet(
    outlines: dict[str, BaseGeometry],
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
        "config_notes": cfg.config_notes,
        "total_rooms_exported": len(df),
        "columns": list(df.columns),
        "room_labels": list(ROOM_NAMES),
    }
    meta_path = cfg.output_dir / f"layouts_{ts}_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return parquet_path


def export_to_csv(
    outlines: dict[str, BaseGeometry],
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
    return csv_path
