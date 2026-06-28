"""Small, testable helpers for the judge-facing demo evidence panels."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import mapping

EXPORT_COLUMNS = [
    "unit_id",
    "input",
    "sample_idx",
    "seed",
    "mode",
    "label",
    "label_idx",
    "geom",
    "wkt",
    "area_m2",
]


@dataclass(frozen=True)
class MetricReport:
    """Metric/report status, with scored metrics left empty when not present."""

    source: str | None
    fid: float | None
    density: float | None
    coverage: float | None
    validity: dict[str, Any]
    n_outlines: int | None
    n_samples: int | None
    candidate_budget: int | None
    checkpoint: str | None
    mode: str | None
    status: str


def generation_mode_code(label: str) -> str:
    """Map the UI label to the canonical generation mode."""

    return "ranked" if label.lower().strip().startswith("ranked") else "raw"


def candidate_budget_from(value: int | float | str | None, default: int = 16) -> int:
    """Return a positive ranked-mode candidate budget."""

    if value is None or value == "":
        return default
    return max(1, int(float(value)))


def layout_to_geojson(layout: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Convert one generated layout to a GeoJSON FeatureCollection."""

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "label": room["label"],
                    "label_idx": int(room["label_idx"]),
                    "area": round(float(room["polygon"].area), 4),
                },
                "geometry": mapping(room["polygon"]),
            }
            for room in (layout or [])
        ],
    }


def build_export_rows(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build WKT/CSV rows using the same columns as ``floorgen.export``.

    Each sample item must include ``unit_id``, ``input``, ``sample_idx``,
    ``seed``, ``mode``, and ``layout``. Layout rooms are the public room records
    returned by ``sample_layouts``.
    """

    rows: list[dict[str, Any]] = []
    for sample in samples:
        for room in sample.get("layout") or []:
            poly = room["polygon"]
            rows.append({
                "unit_id": sample["unit_id"],
                "input": sample["input"],
                "sample_idx": int(sample["sample_idx"]),
                "seed": int(sample["seed"]),
                "mode": sample["mode"],
                "label": room["label"],
                "label_idx": int(room["label_idx"]),
                "geom": poly.wkt,
                "wkt": poly.wkt,
                "area_m2": round(float(poly.area), 6),
            })
    return rows


def discover_metric_reports(root: Path = Path("reports")) -> list[Path]:
    """Return existing metric-like reports, newest first."""

    candidate_paths = [
        root / "post_train" / "post_train_report.json",
        root / "ranked_amd_representative.json",
        root / "raw_amd_representative.json",
        root / "ranked_demo_eval.json",
        root / "ranked_amd_smoke.json",
    ]
    discovered: set[Path] = {path for path in candidate_paths if path.exists()}
    if root.exists():
        discovered.update(path for path in root.rglob("*_eval.json") if path.is_file())
        discovered.update(path for path in root.rglob("*report*.json") if path.is_file())
    return sorted(discovered, key=lambda path: path.stat().st_mtime, reverse=True)


def load_metric_status(root: Path = Path("reports")) -> MetricReport:
    """Load the newest report and expose scored metrics honestly if present."""

    reports = discover_metric_reports(root)
    if not reports:
        return MetricReport(
            source=None,
            fid=None,
            density=None,
            coverage=None,
            validity={},
            n_outlines=None,
            n_samples=None,
            candidate_budget=None,
            checkpoint=None,
            mode=None,
            status="No offline report found.",
        )

    source = reports[0]
    try:
        data = json.loads(source.read_text())
    except Exception as exc:
        return MetricReport(
            source=str(source),
            fid=None,
            density=None,
            coverage=None,
            validity={},
            n_outlines=None,
            n_samples=None,
            candidate_budget=None,
            checkpoint=None,
            mode=None,
            status=f"Could not read report: {exc}",
        )

    return MetricReport(
        source=str(source),
        fid=_find_metric(data, {"fid", "frechet_inception_distance"}),
        density=_find_metric(data, {"density", "prdc_density"}),
        coverage=_find_metric(data, {"coverage", "prdc_coverage"}),
        validity=data.get("validity") if isinstance(data.get("validity"), dict) else {},
        n_outlines=_as_int(data.get("n_outlines")),
        n_samples=_as_int(data.get("n_samples_per_outline") or data.get("n_samples")),
        candidate_budget=_as_int(data.get("candidate_budget")),
        checkpoint=str(data.get("checkpoint")) if data.get("checkpoint") is not None else None,
        mode=str(data.get("mode")) if data.get("mode") is not None else None,
        status="Loaded newest local report.",
    )


def _find_metric(data: Any, names: set[str]) -> float | None:
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in names and isinstance(value, int | float):
                return float(value)
        for value in data.values():
            found = _find_metric(value, names)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_metric(item, names)
            if found is not None:
                return found
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
