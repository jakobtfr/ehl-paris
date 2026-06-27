"""Minimum-rotated-rectangle room representation.

The primary generator emits one token per room slot:
``(cx, cy, w, h, angle, label_idx, presence)``. This module converts between
room polygons and MRR tokens, then applies the deterministic validity-repair
layer. The repair layer may clean and clip generated geometry, but it must not
become a from-scratch partitioner.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..config import (
    MAX_REPAIR_GAP_FRAC,
    MAX_REPAIR_OVERLAP_FRAC,
    MIN_ROOM_AREA_M2,
    ROOM_NAMES,
    SNAP_TOL_M,
)


class RepairRejected(ValueError):
    """Raised when repair would need to invent too much layout geometry."""


@dataclass(frozen=True)
class PartitionAccounting:
    """Area fractions that describe partition validity against an outline."""

    overlap_frac: float
    gap_frac: float
    outside_frac: float


@dataclass
class RoomMRR:
    """One room as a minimum rotated rectangle plus a label."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float
    label_idx: int

    def __post_init__(self) -> None:
        self.cx = float(self.cx)
        self.cy = float(self.cy)
        w = max(float(self.w), 0.0)
        h = max(float(self.h), 0.0)
        angle = float(self.angle)
        if h > w:
            w, h = h, w
            angle += math.pi / 2.0
        self.w = w
        self.h = h
        self.angle = canonical_angle(angle)
        self.label_idx = _canonical_label_idx(self.label_idx)

    @property
    def label(self) -> str:
        return ROOM_NAMES[self.label_idx]

    @property
    def has_finite_geometry(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (self.cx, self.cy, self.w, self.h, self.angle)
        )

    def to_polygon(self) -> Polygon:
        if not self.has_finite_geometry or self.w <= 0 or self.h <= 0:
            return Polygon()
        w2, h2 = self.w / 2.0, self.h / 2.0
        poly = Polygon([(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)])
        poly = affinity.rotate(poly, self.angle, origin=(0, 0), use_radians=True)
        return affinity.translate(poly, xoff=self.cx, yoff=self.cy)


def canonical_angle(angle: float) -> float:
    """Map rectangle orientation to [-pi/2, pi/2)."""

    return ((angle + math.pi / 2.0) % math.pi) - math.pi / 2.0


def wrapped_angle_distance(a: float, b: float) -> float:
    """Smallest absolute angle distance for rectangle orientations modulo pi."""

    return abs(canonical_angle(a - b))


def _canonical_label_idx(label_idx: float) -> int:
    value = float(label_idx)
    if not math.isfinite(value):
        return 0
    return min(max(int(round(value)), 0), len(ROOM_NAMES) - 1)


def polygon_to_mrr(poly: Polygon, label_idx: int) -> RoomMRR:
    """Encode a room polygon as its Shapely minimum rotated rectangle."""

    rect = poly.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:-1]
    if len(coords) != 4:
        c = poly.centroid
        return RoomMRR(c.x, c.y, 0.0, 0.0, 0.0, label_idx)

    edges = []
    for i, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(i + 1) % 4]
        length = math.hypot(x1 - x0, y1 - y0)
        edges.append((length, x1 - x0, y1 - y0))

    # Width is the longer rectangle side. The angle points along width, so the
    # same rectangle has one canonical width/height/angle tuple.
    i = max(range(4), key=lambda idx: edges[idx][0])
    w, dx, dy = edges[i]
    h = edges[(i + 1) % 4][0]
    angle = canonical_angle(math.atan2(dy, dx))

    c = rect.centroid
    return RoomMRR(
        cx=float(c.x),
        cy=float(c.y),
        w=float(max(w, 0.0)),
        h=float(max(h, 0.0)),
        angle=float(angle),
        label_idx=label_idx,
    )


def geometry_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """Area IoU for two Shapely geometries."""

    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(a.intersection(b).area / union)


def encode_decode_iou(poly: Polygon, label_idx: int = 0) -> float:
    """IoU between a polygon and its raw MRR encode/decode reconstruction."""

    if poly.is_empty or poly.area <= 0:
        return 0.0
    return geometry_iou(poly, polygon_to_mrr(poly, label_idx).to_polygon())


def partition_accounting(
    parts: list[tuple[BaseGeometry, int]],
    outline: BaseGeometry,
) -> PartitionAccounting:
    """Compute overlap, gap, and outside-outline fractions for room parts."""

    outline_area = outline.area if outline.area > 0 else 1.0
    polys = [poly for poly, _ in parts if not poly.is_empty and poly.area > 0]
    if not polys:
        return PartitionAccounting(
            overlap_frac=0.0,
            gap_frac=float(outline.area / outline_area),
            outside_frac=0.0,
        )

    union = unary_union(polys)
    overlap = max(sum(poly.area for poly in polys) - union.area, 0.0)
    return PartitionAccounting(
        overlap_frac=float(overlap / outline_area),
        gap_frac=float(outline.difference(union).area / outline_area),
        outside_frac=float(union.difference(outline).area / outline_area),
    )


def mrrs_to_array(mrrs: list[RoomMRR]) -> np.ndarray:
    """Pack MRRs into an (N, 6) array: cx, cy, w, h, angle, label_idx."""

    if not mrrs:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(
        [[m.cx, m.cy, m.w, m.h, m.angle, m.label_idx] for m in mrrs],
        dtype=np.float32,
    )


def array_to_mrrs(arr: np.ndarray) -> list[RoomMRR]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        if arr.shape[0] != 6:
            raise ValueError(f"MRR array must have shape (N, 6), got {arr.shape}")
        arr = arr.reshape(1, 6)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"MRR array must have shape (N, 6), got {arr.shape}")

    return [
        RoomMRR(
            cx=float(r[0]),
            cy=float(r[1]),
            w=float(r[2]),
            h=float(r[3]),
            angle=float(r[4]),
            label_idx=float(r[5]),
        )
        for r in arr
    ]


def repair_partition(
    mrrs: list[RoomMRR],
    outline: Polygon | MultiPolygon,
    *,
    snap_tol: float = SNAP_TOL_M,
    min_area: float = MIN_ROOM_AREA_M2,
    max_gap_frac: float = MAX_REPAIR_GAP_FRAC,
    max_overlap_frac: float = MAX_REPAIR_OVERLAP_FRAC,
    reject_large_repairs: bool = True,
) -> list[tuple[Polygon, int]]:
    """Turn raw MRR tokens into a clean partition of the outline.

    Large overlaps/gaps are rejected because filling them would make the repair
    layer a rule-based partitioner. Small slivers are assigned to the nearest
    accepted room with a deterministic rule.
    """

    if not mrrs:
        return []

    clipped: list[tuple[Polygon, int]] = []
    for room in mrrs:
        inter = room.to_polygon().intersection(outline)
        if inter.is_empty or inter.area <= 0:
            continue
        for part in _iter_polygons(inter):
            if part.area > 0:
                clipped.append((part, room.label_idx))
    if not clipped:
        return []

    outline_area = outline.area if outline.area > 0 else 1.0
    clipped_union = unary_union([poly for poly, _ in clipped])
    overlap_frac = max(sum(poly.area for poly, _ in clipped) - clipped_union.area, 0.0) / outline_area
    if reject_large_repairs and overlap_frac > max_overlap_frac:
        raise RepairRejected(
            f"overlap repair too large: {overlap_frac:.3f} > {max_overlap_frac:.3f}"
        )

    clipped.sort(key=lambda t: t[0].area, reverse=True)
    claimed: BaseGeometry | None = None
    resolved: list[tuple[Polygon, int]] = []
    for poly, label in clipped:
        free = poly if claimed is None else poly.difference(claimed)
        for part in _iter_polygons(free):
            if part.area >= min_area:
                resolved.append((part, label))
        claimed = part_union(claimed, poly)

    if not resolved:
        return []

    covered = unary_union([p for p, _ in resolved])
    gap = outline.difference(covered)
    gap_frac = gap.area / outline_area if not gap.is_empty else 0.0
    if reject_large_repairs and gap_frac > max_gap_frac:
        raise RepairRejected(
            f"gap repair too large: {gap_frac:.3f} > {max_gap_frac:.3f}"
        )

    if not gap.is_empty and gap.area > 0:
        for piece in _iter_polygons(gap):
            if piece.area <= 0:
                continue
            idx = _nearest_room_idx(piece, resolved)
            _merge_room_piece(resolved, idx, piece)

    out: list[tuple[Polygon, int]] = []
    for poly, label in resolved:
        clean = poly.buffer(0).simplify(snap_tol / 2).intersection(outline)
        for part in _iter_polygons(clean):
            if part.area >= min_area and part.is_valid:
                out.append((part, label))
    return out


def part_union(a: BaseGeometry | None, b: BaseGeometry) -> BaseGeometry:
    return b if a is None else unary_union([a, b])


def _iter_polygons(geom: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            yield geom
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield g
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            yield from _iter_polygons(g)


def _merge_room_piece(rooms: list[tuple[Polygon, int]], idx: int, piece: Polygon) -> None:
    poly, label = rooms[idx]
    parts = sorted(
        _iter_polygons(part_union(poly, piece)),
        key=lambda p: (-p.intersection(poly).area, -p.area, p.bounds),
    )
    if not parts:
        return
    rooms[idx] = (parts[0], label)
    rooms.extend((part, label) for part in parts[1:])


def _nearest_room_idx(piece: Polygon, rooms: list[tuple[Polygon, int]]) -> int:
    best, best_d = 0, float("inf")
    for i, (poly, _) in enumerate(rooms):
        d = poly.distance(piece)
        if d < best_d:
            best, best_d = i, d
    return best
