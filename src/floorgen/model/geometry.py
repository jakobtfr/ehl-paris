"""Geometry adapters between processed MSD records and model tensors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from ..data.normalize import PlanTransform, fit_transform, scale_features
from ..data.outline import largest_shell
from ..repr.mrr import RoomMRR, polygon_to_mrr

SCALE_FEATURE_KEYS = (
    "area_m2",
    "bbox_w_m",
    "bbox_h_m",
    "aspect_ratio",
    "perimeter_m",
    "compactness",
)


@dataclass(frozen=True)
class TargetArrays:
    """Padded fixed-slot target tensors before conversion to torch."""

    target_geom: np.ndarray
    target_type: np.ndarray
    present: np.ndarray
    n_truncated: int


@dataclass(frozen=True)
class OutlineConditioning:
    """Numpy conditioning arrays plus the metric inverse transform."""

    outline_xy: np.ndarray
    outline_mask: np.ndarray
    scale: np.ndarray
    transform: PlanTransform


def transform_from_record(record: Mapping[str, Any]) -> PlanTransform:
    raw = record["transform"]
    return PlanTransform(
        centroid_x=float(raw["centroid_x"]),
        centroid_y=float(raw["centroid_y"]),
        scale=float(raw["scale"]),
    )


def scale_feature_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Return stable numeric conditioning scalars in a fixed six-field order."""

    area = max(float(features["area_m2"]), 0.0)
    bbox_w = max(float(features["bbox_w_m"]), 0.0)
    bbox_h = max(float(features["bbox_h_m"]), 0.0)
    aspect = float(features["aspect_ratio"])
    perimeter = max(float(features["perimeter_m"]), 0.0)
    compactness = float(features["compactness"])
    return np.array(
        [
            math.log1p(area),
            math.log1p(bbox_w),
            math.log1p(bbox_h),
            aspect,
            math.log1p(perimeter),
            compactness,
        ],
        dtype=np.float32,
    )


def _single_shell(geom: BaseGeometry) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return largest_shell(geom)
    raise TypeError(f"Expected Polygon or MultiPolygon, got {geom.geom_type}")


def sample_boundary_points(outline: BaseGeometry, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample evenly spaced exterior boundary points from a normalized outline."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")

    xy = np.zeros((n_points, 2), dtype=np.float32)
    mask = np.zeros((n_points,), dtype=bool)
    if outline.is_empty:
        return xy, mask

    shell = _single_shell(outline)
    boundary = shell.exterior
    length = float(boundary.length)
    if length <= 0.0:
        return xy, mask

    for i, distance in enumerate(np.linspace(0.0, length, n_points, endpoint=False)):
        point = boundary.interpolate(float(distance))
        xy[i] = (float(point.x), float(point.y))
    mask[:] = True
    return xy, mask


def outline_conditioning_from_record(
    record: Mapping[str, Any],
    *,
    boundary_points: int,
) -> OutlineConditioning:
    transform = transform_from_record(record)
    outline = wkt.loads(record["outline_wkt"])
    normalized_outline = transform.normalize(outline)
    outline_xy, outline_mask = sample_boundary_points(normalized_outline, boundary_points)
    return OutlineConditioning(
        outline_xy=outline_xy,
        outline_mask=outline_mask,
        scale=scale_feature_vector(record["scale_features"]),
        transform=transform,
    )


def outline_conditioning_from_geometry(
    outline: BaseGeometry,
    *,
    boundary_points: int,
) -> OutlineConditioning:
    if outline.is_empty:
        raise ValueError("outline must be non-empty")
    transform = fit_transform(outline)
    normalized_outline = transform.normalize(outline)
    outline_xy, outline_mask = sample_boundary_points(normalized_outline, boundary_points)
    return OutlineConditioning(
        outline_xy=outline_xy,
        outline_mask=outline_mask,
        scale=scale_feature_vector(asdict(scale_features(outline))),
        transform=transform,
    )


def _room_polygon_from_wkt(value: str) -> Polygon | None:
    geom = wkt.loads(value)
    if geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return largest_shell(geom)
    return None


def record_to_targets(record: Mapping[str, Any], *, k: int) -> TargetArrays:
    """Encode one processed record into fixed-slot normalized MRR targets."""

    if k <= 0:
        raise ValueError("k must be positive")

    transform = transform_from_record(record)
    target_geom = np.zeros((k, 5), dtype=np.float32)
    target_type = np.zeros((k,), dtype=np.int64)
    present = np.zeros((k,), dtype=np.float32)

    rooms = sorted(
        record["rooms"],
        key=lambda room: float(room.get("area_m2", 0.0)),
        reverse=True,
    )
    n_truncated = max(0, len(rooms) - k)
    slot = 0
    for room in rooms:
        if slot >= k:
            break
        poly = _room_polygon_from_wkt(room["wkt"])
        if poly is None or poly.area <= 0:
            continue
        label_idx = int(room["label_idx"])
        normalized_poly = transform.normalize(poly)
        if not isinstance(normalized_poly, Polygon):
            normalized_poly = _single_shell(normalized_poly)
        mrr = polygon_to_mrr(normalized_poly, label_idx)
        if not mrr.has_finite_geometry or mrr.w <= 0.0 or mrr.h <= 0.0:
            continue
        target_geom[slot] = (mrr.cx, mrr.cy, mrr.w, mrr.h, mrr.angle)
        target_type[slot] = mrr.label_idx
        present[slot] = 1.0
        slot += 1

    return TargetArrays(
        target_geom=target_geom,
        target_type=target_type,
        present=present,
        n_truncated=n_truncated,
    )


def decode_mrr_slots(
    geom: np.ndarray,
    type_idx: np.ndarray,
    present: np.ndarray,
    transform: PlanTransform,
    *,
    min_size: float = 1e-3,
) -> list[RoomMRR]:
    """Decode normalized model slots back to metric-space MRRs."""

    geom = np.asarray(geom, dtype=np.float32)
    type_idx = np.asarray(type_idx)
    present = np.asarray(present)
    if geom.ndim != 2 or geom.shape[1] != 5:
        raise ValueError(f"geom must have shape (K, 5), got {geom.shape}")

    mrrs: list[RoomMRR] = []
    for row, label, keep in zip(geom, type_idx, present):
        if not bool(keep) or not np.isfinite(row).all():
            continue
        cx, cy, w, h, angle = (float(v) for v in row)
        w = abs(w)
        h = abs(h)
        if w < min_size or h < min_size:
            continue
        normalized = RoomMRR(cx=cx, cy=cy, w=w, h=h, angle=angle, label_idx=int(label))
        metric_poly = transform.invert(normalized.to_polygon())
        if metric_poly.is_empty or metric_poly.area <= 0:
            continue
        if not isinstance(metric_poly, Polygon):
            metric_poly = _single_shell(metric_poly)
        mrrs.append(polygon_to_mrr(metric_poly, int(label)))
    return mrrs
