"""Judge-facing demo dashboard for outline-conditioned room generation.

The demo calls the public generation API instead of a private demo path. It is
therefore an inspectable view of the same checkpoint, repair, ranking, and
export seams used by the CLIs.
"""

from __future__ import annotations

import copy
import hashlib
import html
import importlib
import json
import math
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
import pandas as pd
from shapely import affinity, wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..baseline import baseline_sample  # noqa: E402
from ..config import ROOM_NAMES, SEED  # noqa: E402
from ..eval.metrics import validity_metrics  # noqa: E402
from ..eval.render import ROOM_COLORS  # noqa: E402
from ..generate import sample_layouts  # noqa: E402
from .evidence import (  # noqa: E402
    EXPORT_COLUMNS,
    build_export_rows,
    candidate_budget_from,
    generation_mode_code,
    layout_to_geojson,
    load_metric_status,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PRESETS = json.loads((Path(__file__).parent / "presets.json").read_text())
DEMO_OUTPUT_DIR = REPO_ROOT / "outputs" / "demo"


CSS = """
:root {
  color-scheme: light;
}
.gradio-container {
  max-width: 1440px !important;
  background: #f8fafc;
  color: #0f172a;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --body-background-fill: #f8fafc;
  --body-text-color: #0f172a;
  --block-background-fill: #ffffff;
  --block-border-color: #e2e8f0;
  --block-label-background-fill: #f8fafc;
  --block-label-text-color: #334155;
  --input-background-fill: #ffffff;
  --input-background-fill-focus: #ffffff;
  --input-border-color: #cbd5e1;
  --input-border-color-focus: #2563eb;
  --input-placeholder-color: #64748b;
  --neutral-50: #f8fafc;
  --neutral-100: #f1f5f9;
  --neutral-200: #e2e8f0;
  --neutral-700: #334155;
  --neutral-800: #1e293b;
  --neutral-900: #0f172a;
}
#app-shell {
  border: 1px solid #e2e8f0;
  background: #ffffff;
  border-radius: 8px;
  padding: 22px;
  box-shadow: 0 20px 45px rgba(15, 23, 42, 0.07);
}
.workspace-grid {
  align-items: flex-start;
}
.control-panel {
  border: 1px solid #e2e8f0;
  background: #ffffff;
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.control-heading {
  border-bottom: 1px solid #e2e8f0;
  margin: -2px 0 14px;
  padding-bottom: 12px;
}
.control-heading strong {
  color: #0f172a;
  display: block;
  font-size: 15px;
  line-height: 1.25;
}
.control-heading span {
  color: #475569;
  display: block;
  font-size: 13px;
  line-height: 1.4;
  margin-top: 4px;
}
.control-panel,
.control-panel * {
  color-scheme: light;
}
.control-panel textarea,
.control-panel input,
.control-panel select {
  background: #ffffff !important;
  color: #0f172a !important;
  border-color: #cbd5e1 !important;
}
.dashboard-title h1 {
  margin: 0 0 4px;
  font-size: 26px;
  color: #0f172a !important;
  letter-spacing: 0;
}
.dashboard-title p {
  margin: 0;
  color: #475569 !important;
  font-size: 14px;
}
.dashboard-title {
  margin-bottom: 26px;
}
.app-kicker {
  color: #2563eb;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .08em;
  margin-bottom: 6px;
  text-transform: uppercase;
}
.run-overview {
  border: 1px solid #e2e8f0;
  background: #ffffff;
  border-radius: 8px;
  padding: 14px;
  margin-bottom: 12px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.run-overview-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
}
.run-title {
  display: block;
  color: #0f172a;
  font-size: 15px;
  font-weight: 750;
  line-height: 1.25;
}
.run-subtitle {
  display: block;
  color: #475569;
  font-size: 13px;
  line-height: 1.45;
  margin-top: 3px;
}
.status-pills {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
  min-width: 220px;
}
.status-pill {
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  background: #f8fafc;
  color: #0f172a;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  font-weight: 650;
  line-height: 1;
  min-height: 28px;
  padding: 7px 10px;
  white-space: nowrap;
}
.status-pill.good {
  background: #ecfdf5;
  border-color: #a7f3d0;
  color: #065f46;
}
.status-pill.warn {
  background: #fff7ed;
  border-color: #fed7aa;
  color: #9a3412;
}
.run-kpis {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.kpi-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #f8fafc;
  min-height: 96px;
  padding: 12px;
}
.kpi-card strong {
  display: block;
  color: #334155;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .02em;
  margin-bottom: 7px;
  text-transform: uppercase;
}
.kpi-card b {
  display: block;
  color: #0f172a;
  font-size: 24px;
  letter-spacing: 0;
  line-height: 1.1;
}
.kpi-card span {
  display: block;
  color: #475569;
  font-size: 12.5px;
  line-height: 1.4;
  margin-top: 6px;
}
.evidence-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 0 0 12px;
}
.evidence-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #ffffff;
  padding: 10px 12px;
  min-height: 78px;
}
.evidence-card strong {
  display: block;
  color: #0f172a;
  font-size: 12px;
  margin-bottom: 5px;
}
.evidence-card span {
  display: block;
  color: #475569;
  font-size: 12px;
  line-height: 1.35;
}
.code-cmd {
  border: 1px solid #334155;
  border-radius: 8px;
  background: #111827;
  color: #f8fafc;
  padding: 10px 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  font-size: 12px;
}
.control-panel label,
.control-panel span {
  color: #0f172a !important;
}
.control-panel span[data-testid="block-info"],
.control-panel .block-info,
.control-panel label > span {
  background: transparent !important;
  border-radius: 0 !important;
  color: #334155 !important;
  font-size: 12px !important;
  font-weight: 650 !important;
  padding: 0 !important;
}
.control-panel .form,
.control-panel .block {
  border-radius: 8px !important;
}
.control-panel [role="radiogroup"] label {
  border: 1px solid #e2e8f0 !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  color: #0f172a !important;
  min-height: 36px !important;
}
.control-panel [role="radiogroup"] label:has(input:checked) {
  border-color: #2563eb !important;
  background: #eff6ff !important;
  box-shadow: 0 0 0 1px #2563eb inset !important;
}
.primary-action button {
  min-height: 44px !important;
  border-radius: 8px !important;
  font-weight: 750 !important;
  width: 100% !important;
}
.plot-panel {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
}
.plot-panel img,
.plot-panel canvas {
  background: #ffffff !important;
}
@media (max-width: 1000px) {
  .run-kpis,
  .evidence-grid {
    grid-template-columns: 1fr;
  }
  .run-overview-top {
    display: block;
  }
  .status-pills {
    justify-content: flex-start;
    margin-top: 10px;
    min-width: 0;
  }
  #app-shell {
    padding: 14px;
  }
}
"""


@dataclass(frozen=True)
class ModelSettings:
    checkpoint_path: str
    device: str
    steps: int
    threshold: float
    candidate_budget: int


@dataclass(frozen=True)
class SampleRun:
    label: str
    mode: str
    layout: list[dict[str, Any]] | None
    error: str | None = None


@dataclass(frozen=True)
class GeneratedGroup:
    name: str
    outline: Polygon
    runs: list[SampleRun]
    ranking: dict[str, Any] | None = None


_APPLIED_MODEL_KEY: tuple[str, str, int, float] | None = None
_MODEL_STATUS: dict[str, Any] | None = None


def _nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _env_int(name: str, default: int, legacy: str | None = None) -> int:
    value = _nonempty_env(name) or (_nonempty_env(legacy) if legacy else None)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float, legacy: str | None = None) -> float:
    value = _nonempty_env(name) or (_nonempty_env(legacy) if legacy else None)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _initial_model_settings() -> ModelSettings:
    return ModelSettings(
        checkpoint_path=_nonempty_env("FLOORGEN_CHECKPOINT") or "",
        device=_nonempty_env("FLOORGEN_DEVICE") or "cpu",
        steps=max(1, _env_int("FLOORGEN_SAMPLE_STEPS", 32, "FLOORGEN_DEMO_STEPS")),
        threshold=min(
            1.0,
            max(0.0, _env_float("FLOORGEN_PRESENCE_THRESHOLD", 0.5, "FLOORGEN_DEMO_THRESHOLD")),
        ),
        candidate_budget=candidate_budget_from(_nonempty_env("FLOORGEN_CANDIDATE_BUDGET"), 16),
    )


INITIAL_SETTINGS = _initial_model_settings()


def _preset_display_label(name: str) -> str:
    try:
        identifier = name.split()[1]
        area = name.split("~", maxsplit=1)[1].split()[0]
        rooms = name.rsplit(",", maxsplit=1)[1].strip().split()[0]
        return f"Apt {identifier} - {area}m2, {rooms}r"
    except (IndexError, ValueError):
        return name.replace("Apartment", "Apt").replace("m\u00b2", "m2")


def _settings_from_controls(
    checkpoint_path: str,
    device: str,
    steps: int | float,
    threshold: int | float,
    candidate_budget: int | float,
) -> ModelSettings:
    return ModelSettings(
        checkpoint_path=(checkpoint_path or "").strip(),
        device=(device or "cpu").strip(),
        steps=max(1, int(steps)),
        threshold=min(1.0, max(0.0, float(threshold))),
        candidate_budget=candidate_budget_from(candidate_budget, INITIAL_SETTINGS.candidate_budget),
    )


def _reset_generate_env_cache(module: Any) -> None:
    if hasattr(module, "_ENV_GENERATOR"):
        module._ENV_GENERATOR = None
    if hasattr(module, "_ENV_GENERATOR_KEY"):
        module._ENV_GENERATOR_KEY = None


def _generator_import_path() -> str:
    module = importlib.import_module("floorgen.generate")
    generator = module.GENERATOR
    generator_module = getattr(generator, "__module__", "")
    generator_name = getattr(generator, "__name__", generator.__class__.__name__)
    return f"{generator_module}.{generator_name}".strip(".")


def _configure_generator(settings: ModelSettings) -> dict[str, Any]:
    """Apply canonical model settings and return explicit backend status."""

    global _APPLIED_MODEL_KEY, _MODEL_STATUS

    key = (
        settings.checkpoint_path,
        settings.device,
        int(settings.steps),
        round(float(settings.threshold), 6),
    )
    if key == _APPLIED_MODEL_KEY and _MODEL_STATUS is not None:
        return _MODEL_STATUS

    module = importlib.import_module("floorgen.generate")
    os.environ["FLOORGEN_DEVICE"] = settings.device
    os.environ["FLOORGEN_SAMPLE_STEPS"] = str(settings.steps)
    os.environ["FLOORGEN_PRESENCE_THRESHOLD"] = str(settings.threshold)
    os.environ["FLOORGEN_CANDIDATE_BUDGET"] = str(settings.candidate_budget)

    if not settings.checkpoint_path:
        os.environ.pop("FLOORGEN_CHECKPOINT", None)
        module.GENERATOR = baseline_sample
        _reset_generate_env_cache(module)
        status = {
            "state": "baseline",
            "backend_label": "Baseline fallback",
            "backend_path": _generator_import_path(),
            "checkpoint_path": "",
            "checkpoint_state": "not configured",
            "message": "No FLOORGEN_CHECKPOINT is set; using the heuristic baseline.",
        }
    else:
        path = Path(settings.checkpoint_path).expanduser()
        os.environ["FLOORGEN_CHECKPOINT"] = str(path)
        if not path.exists():
            module.GENERATOR = baseline_sample
            _reset_generate_env_cache(module)
            status = {
                "state": "missing",
                "backend_label": "Checkpoint missing",
                "backend_path": _generator_import_path(),
                "checkpoint_path": str(path),
                "checkpoint_state": "missing",
                "message": f"FLOORGEN_CHECKPOINT is set but missing: {path}",
            }
        else:
            try:
                from ..model.sampler import load_generator

                module.GENERATOR = load_generator(
                    path,
                    device=settings.device,
                    steps=settings.steps,
                    threshold=settings.threshold,
                )
                _reset_generate_env_cache(module)
                status = {
                    "state": "loaded",
                    "backend_label": "Flow checkpoint sampler",
                    "backend_path": _generator_import_path(),
                    "checkpoint_path": str(path),
                    "checkpoint_state": "loaded",
                    "message": f"Loaded checkpoint sampler: {path.name}",
                }
            except Exception as exc:  # pragma: no cover - depends on torch/checkpoint runtime
                module.GENERATOR = baseline_sample
                _reset_generate_env_cache(module)
                status = {
                    "state": "error",
                    "backend_label": "Checkpoint load error",
                    "backend_path": _generator_import_path(),
                    "checkpoint_path": str(path),
                    "checkpoint_state": "error",
                    "message": f"Checkpoint load failed: {exc}",
                }

    status.update({
        "device": settings.device,
        "steps": int(settings.steps),
        "threshold": float(settings.threshold),
        "candidate_budget": int(settings.candidate_budget),
    })
    _APPLIED_MODEL_KEY = key
    _MODEL_STATUS = status
    return status


CHECKPOINT_STATUS = _configure_generator(INITIAL_SETTINGS)


def _largest_polygon(geom: BaseGeometry) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    raise ValueError("outline must be a Polygon or MultiPolygon")


def _parse_outline(preset_name: str, custom_wkt: str | None) -> Polygon:
    custom = (custom_wkt or "").strip()
    src = custom if custom else PRESETS[preset_name]
    try:
        geom = wkt.loads(src)
    except Exception as exc:
        raise gr.Error(f"Could not parse outline WKT: {exc}") from exc

    outline = _largest_polygon(geom)
    if outline.is_empty or not outline.is_valid:
        repaired = outline.buffer(0)
        if repaired.is_empty or not repaired.is_valid:
            raise gr.Error("Outline must be a valid non-empty polygon.")
        outline = _largest_polygon(repaired)
    return outline


def _near_twin_outline(outline: Polygon) -> Polygon:
    """Create a small aspect/area perturbation for input-sensitivity review."""

    variant = affinity.scale(outline, xfact=1.055, yfact=0.965, origin="centroid")
    if not variant.is_valid:
        variant = variant.buffer(0)
    return _largest_polygon(variant)


def _label_counts(layout: list[dict[str, Any]] | None) -> str:
    counts = Counter(str(room["label"]) for room in (layout or []))
    return ", ".join(
        f"{label} x{count}" if count > 1 else label
        for label, count in counts.most_common()
    )


def _layout_signature(layout: list[dict[str, Any]] | None, outline: Polygon) -> str:
    if not layout:
        return ""
    pieces = []
    outline_area = max(outline.area, 1e-9)
    for room in sorted(layout, key=lambda r: (str(r["label"]), -float(r["polygon"].area))):
        poly = room["polygon"]
        c = poly.centroid
        pieces.append(
            f"{room['label_idx']}:{poly.area / outline_area:.3f}:{c.x:.2f}:{c.y:.2f}"
        )
    return hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()[:8]


def _area_profile(layout: list[dict[str, Any]] | None, outline: Polygon) -> dict[str, float]:
    outline_area = max(outline.area, 1e-9)
    profile: dict[str, float] = {}
    for room in layout or []:
        profile[str(room["label"])] = profile.get(str(room["label"]), 0.0) + (
            float(room["polygon"].area) / outline_area
        )
    return profile


def _mean_profile_delta(profiles: list[dict[str, float]]) -> float:
    if len(profiles) < 2:
        return 0.0
    labels = set().union(*(profile.keys() for profile in profiles))
    total = 0.0
    pairs = 0
    for i, left in enumerate(profiles):
        for right in profiles[i + 1:]:
            total += sum(abs(left.get(label, 0.0) - right.get(label, 0.0)) for label in labels) / 2
            pairs += 1
    return total / max(pairs, 1)


def _rows_for_group(group: GeneratedGroup) -> list[dict[str, Any]]:
    rows = []
    for idx, run in enumerate(group.runs, start=1):
        rooms = [(room["polygon"], int(room["label_idx"])) for room in (run.layout or [])]
        metrics = validity_metrics(rooms, group.outline)
        rows.append({
            "input": group.name,
            "sample": run.label or f"S{idx}",
            "mode": run.mode,
            "status": "failed" if run.error else "ok",
            "rooms": len(run.layout or []),
            "labels": _label_counts(run.layout),
            "outside_pct": round(metrics["outside_frac"] * 100, 3),
            "overlap_pct": round(metrics["overlap_frac"] * 100, 3),
            "gap_pct": round(metrics["gap_frac"] * 100, 3),
            "invalid_pct": round(metrics["invalid_rate"] * 100, 3),
            "partition_pct": round((1 - metrics["gap_frac"]) * 100, 2),
            "signature": _layout_signature(run.layout, group.outline),
            "error": run.error or "",
        })
    return rows


def _square_limits(outline: Polygon) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = outline.bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    pad = max(width, height) * 0.08
    extent = max(width, height) + 2 * pad
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    return cx - extent / 2, cx + extent / 2, cy - extent / 2, cy + extent / 2


def _room_abbrev(label: str) -> str:
    aliases = {
        "Bedroom": "Bed",
        "Livingroom": "Live",
        "Kitchen": "Kit",
        "Dining": "Din",
        "Corridor": "Hall",
        "Stairs": "Stair",
        "Storeroom": "Store",
        "Bathroom": "Bath",
        "Balcony": "Bal",
        "Structure": "Core",
    }
    return aliases.get(label, label[:4])


def _draw_panel(
    ax: Any,
    outline: Polygon,
    layout: list[dict[str, Any]] | None,
    title: str,
    error: str | None = None,
) -> None:
    ax.set_facecolor("#ffffff")
    x0, x1, y0, y1 = _square_limits(outline)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, color="#172033", pad=8)

    ox, oy = outline.exterior.xy
    if layout is None:
        ax.fill(ox, oy, facecolor="#eef2f7", edgecolor="#111827", linewidth=1.8)
        ax.plot(ox, oy, color="#111827", linewidth=1.8)
        if error:
            ax.text(
                0.5,
                0.5,
                f"Strict repair rejected\n{error[:96]}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="#7c2d12",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7ed", "ec": "#fdba74"},
                wrap=True,
            )
        return

    outline_area = max(outline.area, 1e-9)
    for room in sorted(layout, key=lambda r: float(r["polygon"].area), reverse=True):
        poly = room["polygon"]
        color = tuple(channel / 255 for channel in ROOM_COLORS.get(room["label"], (90, 90, 90)))
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color=color, ec="#101820", lw=0.65, alpha=0.96)
        if poly.area / outline_area >= 0.035:
            c = poly.representative_point()
            ax.text(
                c.x,
                c.y,
                _room_abbrev(str(room["label"])),
                ha="center",
                va="center",
                fontsize=6.5,
                color="#0f172a",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.65},
            )

    ax.plot(ox, oy, color="#111827", linewidth=1.6)


def _draw_showcase(groups: list[GeneratedGroup], title: str) -> Any:
    panels: list[tuple[str, Polygon, list[dict[str, Any]] | None, str | None]] = []
    for group in groups:
        panels.append((f"{group.name} outline", group.outline, None, None))
        for run in group.runs:
            status = "failed" if run.error else f"{len(run.layout or [])} rooms"
            panels.append((f"{group.name} {run.label} | {status}", group.outline, run.layout, run.error))

    cols = min(3, max(1, len(panels)))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.55 * rows), squeeze=False)
    fig.patch.set_facecolor("#f6f7fb")

    for ax, panel in zip(axes.ravel(), panels, strict=False):
        panel_title, outline, layout, error = panel
        _draw_panel(ax, outline, layout, panel_title, error)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#172033", y=0.995)
    fig.tight_layout(pad=1.2)
    return fig


def _postprocess_status(mode: str, candidate_budget: int) -> str:
    if mode == "raw":
        return "Raw: sampler output plus strict deterministic repair; ranking provenance is not applicable."
    return (
        "Ranked: floorgen.postprocess.rank_samples generates "
        f"{candidate_budget} candidates, records repair-aware scores, and selects diverse layouts."
    )


def _summary_html(
    groups: list[GeneratedGroup],
    rows: list[dict[str, Any]],
    generation_mode: str,
    model_status: dict[str, Any],
    settings: ModelSettings,
) -> str:
    signatures = {row["signature"] for row in rows if row["signature"]}
    room_counts = [int(row["rooms"]) for row in rows if row["status"] == "ok"]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    pass_count = sum(
        1
        for row in ok_rows
        if row["outside_pct"] <= 2.0
        and row["overlap_pct"] <= 2.0
        and row["invalid_pct"] == 0.0
        and row["gap_pct"] <= 12.0
    )
    mean_gap = sum(float(row["gap_pct"]) for row in ok_rows) / max(len(ok_rows), 1)
    mean_outside = sum(float(row["outside_pct"]) for row in ok_rows) / max(len(ok_rows), 1)
    mean_overlap = sum(float(row["overlap_pct"]) for row in ok_rows) / max(len(ok_rows), 1)
    profiles = [
        _area_profile(run.layout, group.outline)
        for group in groups
        for run in group.runs
        if run.layout
    ]
    profile_delta = _mean_profile_delta(profiles)
    mode = generation_mode_code(generation_mode)
    checkpoint_note = model_status["message"]
    room_range = (
        f"{min(room_counts)}-{max(room_counts)}"
        if room_counts and min(room_counts) != max(room_counts)
        else str(room_counts[0] if room_counts else 0)
    )
    metric_status = load_metric_status(REPO_ROOT / "reports")
    metric_source = Path(metric_status.source).name if metric_status.source else "not run"
    backend_label = str(model_status["backend_label"])
    backend_class = "good" if model_status["state"] == "loaded" else "warn"
    checkpoint_name = (
        Path(str(model_status.get("checkpoint_path") or "")).name
        if model_status.get("checkpoint_path")
        else "baseline"
    )
    metric_note = (
        f"Metrics: FID {_fmt_metric(metric_status.fid)}, "
        f"density {_fmt_metric(metric_status.density)}, "
        f"coverage {_fmt_metric(metric_status.coverage)}"
    )
    mode_note = (
        "ranked candidate selection"
        if mode == "ranked"
        else "raw strict repair"
    )

    return f"""
<div class="run-overview">
  <div class="run-overview-top">
    <div>
      <span class="run-title">{html.escape(backend_label)} · {html.escape(mode_note)}</span>
      <span class="run-subtitle">{html.escape(checkpoint_note)} The generated rooms below are vector polygons; detailed provenance lives in the tabs.</span>
    </div>
    <div class="status-pills">
      <span class="status-pill {backend_class}">{html.escape(checkpoint_name)}</span>
      <span class="status-pill">{html.escape(settings.device)} · {settings.steps} steps</span>
      <span class="status-pill">{settings.candidate_budget} candidates</span>
      <span class="status-pill">{html.escape(metric_source)}</span>
    </div>
  </div>
  <div class="run-kpis">
    <div class="kpi-card">
      <strong>Validity</strong>
      <b>{pass_count}/{len(ok_rows)}</b>
      <span>Samples pass containment, overlap, gap, and invalid-geometry checks.</span>
    </div>
    <div class="kpi-card">
      <strong>Diversity</strong>
      <b>{len(signatures)}</b>
      <span>Distinct layout signatures; area-profile delta {profile_delta:.1%}.</span>
    </div>
    <div class="kpi-card">
      <strong>Geometry</strong>
      <b>{mean_gap:.2f}%</b>
      <span>Mean gap. Outside {mean_outside:.2f}%, overlap {mean_overlap:.2f}%, rooms {room_range}. {html.escape(metric_note)}.</span>
    </div>
  </div>
</div>
"""


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _ranking_payloads(groups: list[GeneratedGroup]) -> list[dict[str, Any]]:
    payloads = []
    for group in groups:
        if group.ranking:
            payloads.append({
                "input": group.name,
                "ranking": group.ranking,
            })
    return payloads


def _ranking_table(groups: list[GeneratedGroup]) -> pd.DataFrame:
    rows = []
    for group in groups:
        ranking = group.ranking or {}
        selected = set(ranking.get("selected_indices") or [])
        for candidate in ranking.get("candidates") or []:
            raw = candidate.get("raw_metrics") or {}
            repaired = candidate.get("repaired_metrics") or {}
            rows.append({
                "input": group.name,
                "candidate": candidate.get("index"),
                "selected": candidate.get("index") in selected,
                "accepted": candidate.get("accepted"),
                "score": candidate.get("score"),
                "repair_mode": candidate.get("repair_mode"),
                "raw_outside_pct": round(float(raw.get("outside_frac", 0.0)) * 100, 3),
                "raw_overlap_pct": round(float(raw.get("overlap_frac", 0.0)) * 100, 3),
                "raw_gap_pct": round(float(raw.get("gap_frac", 0.0)) * 100, 3),
                "repaired_outside_pct": round(float(repaired.get("outside_frac", 0.0)) * 100, 3),
                "repaired_overlap_pct": round(float(repaired.get("overlap_frac", 0.0)) * 100, 3),
                "repaired_gap_pct": round(float(repaired.get("gap_frac", 0.0)) * 100, 3),
                "rejection_reason": candidate.get("rejection_reason") or "",
                "signature": candidate.get("signature"),
            })
    return pd.DataFrame(rows)


def _ranking_summary_html(groups: list[GeneratedGroup], mode: str) -> str:
    payloads = _ranking_payloads(groups)
    if mode != "ranked":
        return """
<div class="evidence-grid">
  <div class="evidence-card"><strong>Ranking</strong><span>Not applicable for raw mode.</span></div>
  <div class="evidence-card"><strong>Raw boundary</strong><span>Raw mode shows strict-repair success or failure without candidate ranking.</span></div>
</div>
"""
    if not payloads:
        return """
<div class="evidence-grid">
  <div class="evidence-card"><strong>Ranking</strong><span>No ranking provenance was captured for this run.</span></div>
</div>
"""

    generated = sum(int(item["ranking"].get("generated_count") or 0) for item in payloads)
    accepted = sum(int(item["ranking"].get("accepted_count") or 0) for item in payloads)
    rejected = sum(int(item["ranking"].get("rejected_count") or 0) for item in payloads)
    candidates = [
        candidate
        for item in payloads
        for candidate in item["ranking"].get("candidates", [])
    ]
    strict_accepted = sum(
        1 for candidate in candidates
        if candidate.get("accepted") and candidate.get("repair_mode") == "strict"
    )
    permissive = sum(
        1 for candidate in candidates
        if candidate.get("accepted") and candidate.get("repair_mode") == "permissive"
    )
    signatures = [
        signature
        for item in payloads
        for signature in item["ranking"].get("selected_signatures", [])
    ]
    signature_text = ", ".join(signatures[:8]) if signatures else "none"
    budget = payloads[-1]["ranking"].get("candidate_budget", "n/a")

    return f"""
<div class="evidence-grid">
  <div class="evidence-card"><strong>Candidate budget</strong><span>{budget} per ranked call; {generated} total candidates generated here.</span></div>
  <div class="evidence-card"><strong>Strict accepted</strong><span>{strict_accepted} candidates passed strict repair without permissive fallback.</span></div>
  <div class="evidence-card"><strong>Permissive repaired</strong><span>{permissive} accepted candidates used documented permissive repair and are penalized.</span></div>
  <div class="evidence-card"><strong>Rejected</strong><span>{rejected} candidates rejected; {accepted} candidates accepted before diversity selection.</span></div>
</div>
<div class="evidence-card"><strong>Selected diverse signatures</strong><span>{html.escape(signature_text)}</span></div>
"""


def _successful_samples(
    groups: list[GeneratedGroup],
    preset_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    samples = []
    sample_idx = 0
    for group in groups:
        for run in group.runs:
            if run.layout:
                samples.append({
                    "unit_id": preset_name,
                    "input": group.name,
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "mode": run.mode,
                    "layout": run.layout,
                })
                sample_idx += 1
    return samples


def _first_successful_layout(groups: list[GeneratedGroup]) -> list[dict[str, Any]] | None:
    for group in groups:
        for run in group.runs:
            if run.layout:
                return run.layout
    return None


def _write_download_files(
    geojson_text: str,
    export_df: pd.DataFrame,
    provenance_text: str,
) -> tuple[str, str, str]:
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geojson_path = DEMO_OUTPUT_DIR / "current_sample.geojson"
    csv_path = DEMO_OUTPUT_DIR / "current_run_vectors.csv"
    provenance_path = DEMO_OUTPUT_DIR / "current_run_provenance.json"
    geojson_path.write_text(geojson_text)
    export_df.to_csv(csv_path, index=False)
    provenance_path.write_text(provenance_text)
    return str(geojson_path), str(csv_path), str(provenance_path)


def _provenance_json(
    groups: list[GeneratedGroup],
    rows: list[dict[str, Any]],
    preset_name: str,
    showcase_mode: str,
    generation_mode: str,
    settings: ModelSettings,
    model_status: dict[str, Any],
    seed: int,
) -> str:
    metric_status = load_metric_status(REPO_ROOT / "reports")
    payload = {
        "preset": preset_name,
        "showcase_mode": showcase_mode,
        "seed": int(seed),
        "generation_mode": generation_mode,
        "mode": generation_mode_code(generation_mode),
        "model_settings": {
            "checkpoint_path": settings.checkpoint_path,
            "checkpoint_state": model_status["checkpoint_state"],
            "device": settings.device,
            "steps": settings.steps,
            "presence_threshold": settings.threshold,
            "candidate_budget": settings.candidate_budget,
            "backend_label": model_status["backend_label"],
            "backend_import_path": model_status["backend_path"],
        },
        "git_sha": _git_sha(),
        "post_processing_status": _postprocess_status(
            generation_mode_code(generation_mode),
            settings.candidate_budget,
        ),
        "room_taxonomy": list(ROOM_NAMES),
        "metric_report": {
            "source": metric_status.source,
            "fid": metric_status.fid,
            "density": metric_status.density,
            "coverage": metric_status.coverage,
            "status": metric_status.status,
        },
        "export_schema": list(EXPORT_COLUMNS),
        "ranking": _ranking_payloads(groups),
        "groups": [
            {
                "name": group.name,
                "outline_area": round(float(group.outline.area), 4),
                "outline_perimeter": round(float(group.outline.length), 4),
                "bounds": [round(float(v), 4) for v in group.outline.bounds],
                "n_runs": len(group.runs),
            }
            for group in groups
        ],
        "sample_metrics": rows,
    }
    return json.dumps(payload, indent=2)


def _sample_runs(
    outline: Polygon,
    *,
    seed: int,
    n_samples: int,
    mode: str,
    candidate_budget: int,
    label_prefix: str = "S",
) -> tuple[list[SampleRun], dict[str, Any] | None]:
    try:
        layouts = sample_layouts(
            outline,
            seed=seed,
            n_samples=n_samples,
            mode=mode,
            candidate_budget=candidate_budget,
        )
    except Exception as exc:
        return [SampleRun(label=f"{label_prefix}1", mode=mode, layout=None, error=str(exc))], None

    generate_module = importlib.import_module("floorgen.generate")
    ranking = (
        copy.deepcopy(generate_module.LAST_RANKING_PROVENANCE)
        if mode == "ranked" and generate_module.LAST_RANKING_PROVENANCE
        else None
    )
    return [
        SampleRun(label=f"{label_prefix}{idx}", mode=mode, layout=layout)
        for idx, layout in enumerate(layouts, start=1)
    ], ranking


def _build_groups(
    outline: Polygon,
    showcase_mode: str,
    mode: str,
    n_samples: int,
    seed: int,
    candidate_budget: int,
) -> tuple[list[GeneratedGroup], str]:
    if showcase_mode.startswith("Near-twin"):
        per_outline = max(1, min(n_samples, 3))
        twin = _near_twin_outline(outline)
        selected_runs, selected_ranking = _sample_runs(
            outline,
            seed=seed,
            n_samples=per_outline,
            mode=mode,
            candidate_budget=candidate_budget,
        )
        twin_runs, twin_ranking = _sample_runs(
            twin,
            seed=seed + 1009,
            n_samples=per_outline,
            mode=mode,
            candidate_budget=candidate_budget,
        )
        return [
            GeneratedGroup("Selected", outline, selected_runs, selected_ranking),
            GeneratedGroup("Near twin", twin, twin_runs, twin_ranking),
        ], "Similar inputs, independently sampled room layouts"

    if showcase_mode.startswith("Raw vs ranked"):
        raw_runs, _raw_ranking = _sample_runs(
            outline,
            seed=seed,
            n_samples=1,
            mode="raw",
            candidate_budget=candidate_budget,
            label_prefix="Raw ",
        )
        ranked_runs, ranked_ranking = _sample_runs(
            outline,
            seed=seed,
            n_samples=1,
            mode="ranked",
            candidate_budget=candidate_budget,
            label_prefix="Ranked ",
        )
        return [
            GeneratedGroup("Selected", outline, raw_runs + ranked_runs, ranked_ranking)
        ], "Raw checkpoint sample vs ranked/post-processed selection"

    n = 1 if showcase_mode.startswith("Single") else n_samples
    runs, ranking = _sample_runs(
        outline,
        seed=seed,
        n_samples=n,
        mode=mode,
        candidate_budget=candidate_budget,
    )
    title = "Single sample inspection" if n == 1 else "Same outline, multiple plausible room layouts"
    return [GeneratedGroup("Selected", outline, runs, ranking)], title


def _fmt_metric(value: float | None) -> str:
    return "not run" if value is None else f"{value:.4g}"


def _metrics_html(settings: ModelSettings) -> str:
    report = load_metric_status(REPO_ROOT / "reports")
    source = html.escape(Path(report.source).name if report.source else "none")
    validity = report.validity or {}
    validity_text = ", ".join(
        f"{html.escape(str(key))}={value}"
        for key, value in validity.items()
    ) or "validity means not available"
    command = (
        "uv run --extra train python scripts/evaluate.py --demo "
        f"--mode ranked --candidate-budget {settings.candidate_budget} --n-samples 2 "
        "--output reports/ranked_demo_eval.json"
    )
    if settings.checkpoint_path:
        command = (
            "uv run --extra train python scripts/evaluate.py --demo "
            f"--checkpoint {settings.checkpoint_path} --device {settings.device} "
            f"--steps {settings.steps} --threshold {settings.threshold:g} "
            f"--mode ranked --candidate-budget {settings.candidate_budget} --n-samples 2 "
            "--output reports/ranked_demo_eval.json"
        )

    return f"""
<div class="evidence-grid">
  <div class="evidence-card"><strong>FID</strong><span>{_fmt_metric(report.fid)}</span></div>
  <div class="evidence-card"><strong>Density</strong><span>{_fmt_metric(report.density)}</span></div>
  <div class="evidence-card"><strong>Coverage</strong><span>{_fmt_metric(report.coverage)}</span></div>
  <div class="evidence-card"><strong>Report source</strong><span>{source}<br>{html.escape(report.status)}</span></div>
</div>
<div class="evidence-grid">
  <div class="evidence-card"><strong>Checkpoint</strong><span>{html.escape(str(report.checkpoint or "not recorded"))}</span></div>
  <div class="evidence-card"><strong>Mode</strong><span>{html.escape(str(report.mode or "not recorded"))}</span></div>
  <div class="evidence-card"><strong>Outlines / samples</strong><span>{report.n_outlines or "n/a"} outlines, {report.n_samples or "n/a"} samples per outline</span></div>
  <div class="evidence-card"><strong>Candidate budget</strong><span>{report.candidate_budget or "n/a"}</span></div>
</div>
<div class="evidence-card"><strong>Validity means</strong><span>{html.escape(validity_text)}</span></div>
<p>FID, density, and coverage are shown only when an offline report contains those values. Validity-only reports do not imply official challenge scores.</p>
<pre class="code-cmd">{html.escape(command)}</pre>
"""


MODEL_MARKDOWN = """
### Model / Checkpoint Story

- Input: a single apartment outline polygon.
- Representation: fixed MRR room tokens `(cx, cy, w, h, angle, type, presence)`.
- Model: Transformer conditional flow matching over outline boundary tokens and room slots.
- Conditioning: 128 boundary points plus scale/shape features.
- Output: typed vector room polygons in the original coordinate space.
- Repair boundary: deterministic validity repair clips rooms to the outline, resolves overlaps, and fills small slivers.
- Ranked mode: candidate sampling plus repair-aware scoring and diversity selection. It is documented test-time compute, not raw model quality.

Primary checkpoint: `checkpoints/flow-transformer-amd-862d422.pt`.

Training metadata from the repo docs: AMD Instinct MI300X / ROCm-HIP through PyTorch's `cuda` API, `d_model=512`, 4 layers, 8 heads, `K=24`, 128 boundary points, 67 epochs, 4,212 optimizer steps, batch size 256, one-hour cap.

Known model boundary: raw checkpoint outputs are finite but still overlap heavily enough that strict repair often rejects them. Ranked/post-processed mode fixes representative geometry validity through documented candidate generation, repair-aware scoring, and diversity selection.
"""


PITCH_MARKDOWN = """
### 5-Minute Demo Flow

1. Problem: generate labelled room polygons from the outline only.
2. Method: direct vector MRR tokens with a Transformer flow-matching sampler.
3. Validity: deterministic repair preserves outline containment and reports residual outside, overlap, gap, and invalid rates.
4. Ranking: documented test-time compute samples multiple candidates and selects valid, diverse repaired layouts.
5. Evidence: same-outline diversity, near-twin sensitivity, raw-vs-ranked comparison, validity table, and vector exports.
6. Metrics: FID, density, and coverage are shown only when an offline report exists; otherwise this dashboard says `not run`.
7. Provenance: checkpoint, device, steps, threshold, candidate budget, seed, git SHA, ranking data, and export schema are recorded.
8. Limitations: raw overlap, label skew, MRR compression, and no fabricated official scores.
"""


LIMITATIONS_MARKDOWN = """
### Known Limitations

- Baseline fallback is still available for contract smoke tests. It is clearly labelled and should not be confused with the trained flow checkpoint.
- Raw AMD checkpoint samples often fail strict repair because generated room slots overlap too much.
- Ranked/post-processed mode is honest test-time compute: model candidates plus deterministic repair-aware scoring, not raw sampler quality.
- Current reports in this repo are validity/reporting artifacts unless they contain explicit FID, density, and coverage fields.
- Semantic label mix remains skewed in representative ranked runs, especially toward bedrooms and living rooms.
- MRR tokens compress irregular rooms; full corner-sequence tokens remain a future fidelity improvement.
"""


def generate_showcase(
    preset_name: str,
    custom_wkt: str,
    showcase_mode: str,
    generation_mode: str,
    n_samples: int,
    seed: int | float,
    candidate_budget: int | float,
    checkpoint_path: str,
    device: str,
    steps: int | float,
    threshold: int | float,
) -> tuple[Any, str, pd.DataFrame, str, pd.DataFrame, str, str, pd.DataFrame, str, str, str, str, str]:
    settings = _settings_from_controls(checkpoint_path, device, steps, threshold, candidate_budget)
    model_status = _configure_generator(settings)
    outline = _parse_outline(preset_name, custom_wkt)
    seed_int = int(seed)
    n = max(1, min(int(n_samples), 6))
    mode = generation_mode_code(generation_mode)

    if model_status["state"] in {"missing", "error"}:
        groups = [GeneratedGroup("Selected", outline, [])]
        title = "Model configuration needs attention"
    else:
        groups, title = _build_groups(
            outline,
            showcase_mode,
            mode,
            n,
            seed_int,
            settings.candidate_budget,
        )

    rows = [row for group in groups for row in _rows_for_group(group)]
    table = pd.DataFrame(rows)
    fig = _draw_showcase(groups, title)
    summary = _summary_html(groups, rows, generation_mode, model_status, settings)

    first_layout = _first_successful_layout(groups)
    geojson_text = json.dumps(layout_to_geojson(first_layout), indent=2)
    samples = _successful_samples(groups, preset_name, seed_int)
    export_rows = build_export_rows(samples)
    export_df = pd.DataFrame(export_rows, columns=EXPORT_COLUMNS)
    ranking_table = _ranking_table(groups)
    ranking_json = json.dumps(_ranking_payloads(groups), indent=2)
    ranking_summary = _ranking_summary_html(groups, mode)
    provenance = _provenance_json(
        groups,
        rows,
        preset_name,
        showcase_mode,
        generation_mode,
        settings,
        model_status,
        seed_int,
    )
    geojson_file, csv_file, provenance_file = _write_download_files(
        geojson_text,
        export_df,
        provenance,
    )
    metrics = _metrics_html(settings)
    return (
        fig,
        summary,
        table,
        geojson_text,
        export_df,
        ranking_summary,
        ranking_table,
        ranking_json,
        provenance,
        metrics,
        geojson_file,
        csv_file,
        provenance_file,
    )


def build_demo() -> gr.Blocks:
    preset_names = list(PRESETS)
    preset_choices = [(_preset_display_label(name), name) for name in preset_names]
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(
        title="floorgen - Davis AI interior layout generator",
        theme=theme,
        css=CSS,
    ) as demo:
        with gr.Column(elem_id="app-shell"):
            gr.HTML(
                """
                <div class="dashboard-title">
                  <div class="app-kicker">Davis AI / TUM.ai Hackathon</div>
                  <h1>floorgen</h1>
                  <p>Outline-conditioned vector room generation with ranked checkpoint evidence.</p>
                </div>
                """
            )
            with gr.Row(elem_classes=["workspace-grid"]):
                with gr.Column(scale=1, elem_classes=["control-panel"]):
                    gr.HTML(
                        """
                        <div class="control-heading">
                          <strong>Generate a judge-ready layout</strong>
                          <span>Start with a preset outline; tune only if you need a deeper inspection.</span>
                        </div>
                        """
                    )
                    preset = gr.Dropdown(
                        choices=preset_choices,
                        value=preset_names[0],
                        label="MSD outline preset",
                    )
                    showcase_mode = gr.Dropdown(
                        choices=[
                            "Same outline diversity",
                            "Near-twin input sensitivity",
                            "Raw vs ranked comparison",
                            "Single sample inspect",
                        ],
                        value="Same outline diversity",
                        label="Showcase",
                    )
                    go = gr.Button("Generate layouts", variant="primary", elem_classes=["primary-action"])

                    with gr.Accordion("Run tuning", open=False):
                        generation_mode = gr.Dropdown(
                            choices=[
                                "Raw samples",
                                "Ranked/post-processed",
                            ],
                            value="Ranked/post-processed",
                            label="Generation mode",
                        )
                        n_samples = gr.Slider(1, 6, value=3, step=1, label="Samples")
                        seed = gr.Number(value=SEED, label="Seed", precision=0)
                        candidate_budget = gr.Slider(
                            1,
                            32,
                            value=INITIAL_SETTINGS.candidate_budget,
                            step=1,
                            label="Candidate budget",
                        )
                    with gr.Accordion("Custom outline", open=False):
                        custom = gr.Textbox(
                            label="WKT polygon",
                            placeholder="POLYGON ((0 0, 10 0, 10 8, 0 8, 0 0))",
                            lines=3,
                        )
                    with gr.Accordion("Advanced model settings", open=False):
                        checkpoint_path = gr.Textbox(
                            value=INITIAL_SETTINGS.checkpoint_path,
                            label="FLOORGEN_CHECKPOINT",
                            placeholder="checkpoints/flow-transformer-amd-862d422.pt",
                        )
                        device = gr.Dropdown(
                            choices=["cpu", "mps", "cuda"],
                            value=INITIAL_SETTINGS.device,
                            label="FLOORGEN_DEVICE",
                            allow_custom_value=True,
                        )
                        steps = gr.Slider(
                            1,
                            64,
                            value=INITIAL_SETTINGS.steps,
                            step=1,
                            label="FLOORGEN_SAMPLE_STEPS",
                        )
                        threshold = gr.Slider(
                            0.0,
                            1.0,
                            value=INITIAL_SETTINGS.threshold,
                            step=0.05,
                            label="FLOORGEN_PRESENCE_THRESHOLD",
                        )

                with gr.Column(scale=3):
                    summary = gr.HTML(label="Judge summary", show_label=False)
                    plot = gr.Plot(label="Layout showcase", show_label=False, elem_classes=["plot-panel"])

            with gr.Tabs():
                with gr.Tab("Checks"):
                    table = gr.Dataframe(
                        label="Geometry and diversity checks",
                        interactive=False,
                        wrap=True,
                    )
                    ranking_summary = gr.HTML(label="Ranked post-processing summary")
                    ranking_table = gr.Dataframe(
                        label="Candidate repair, scoring, and selection data",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("Export"):
                    geojson = gr.Code(
                        label="First successful sample (GeoJSON FeatureCollection)",
                        language="json",
                        lines=18,
                    )
                    export_df = gr.Dataframe(
                        label="Current run WKT/CSV rows",
                        interactive=False,
                        wrap=True,
                    )
                    with gr.Row():
                        geojson_download = gr.File(label="GeoJSON download")
                        csv_download = gr.File(label="CSV download")
                        provenance_download = gr.File(label="Provenance download")
                with gr.Tab("Provenance"):
                    metrics = gr.HTML(label="Metric/report status")
                    provenance = gr.Code(
                        label="Generation metadata",
                        language="json",
                        lines=20,
                    )
                    ranking_json = gr.Code(
                        label="Full ranking provenance",
                        language="json",
                        lines=18,
                    )
                with gr.Tab("Notes"):
                    gr.Markdown(MODEL_MARKDOWN)
                    gr.Markdown(PITCH_MARKDOWN)
                    gr.Markdown(LIMITATIONS_MARKDOWN)

        inputs = [
            preset,
            custom,
            showcase_mode,
            generation_mode,
            n_samples,
            seed,
            candidate_budget,
            checkpoint_path,
            device,
            steps,
            threshold,
        ]
        outputs = [
            plot,
            summary,
            table,
            geojson,
            export_df,
            ranking_summary,
            ranking_table,
            ranking_json,
            provenance,
            metrics,
            geojson_download,
            csv_download,
            provenance_download,
        ]
        go.click(generate_showcase, inputs, outputs)
        demo.load(generate_showcase, inputs, outputs)
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
