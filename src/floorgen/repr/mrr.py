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
from shapely.geometry import MultiPolygon, Polygon
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


@dataclass
class RoomMRR:
    """One room as a minimum rotated rectangle plus a label."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float
    label_idx: int

    @property
    def label(self) -> str:
        return ROOM_NAMES[self.label_idx]

    def to_polygon(self) -> Polygon:
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


def mrrs_to_array(mrrs: list[RoomMRR]) -> np.ndarray:
    """Pack MRRs into an (N, 6) array: cx, cy, w, h, angle, label_idx."""

    if not mrrs:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(
        [[m.cx, m.cy, m.w, m.h, m.angle, m.label_idx] for m in mrrs],
        dtype=np.float32,
    )


def array_to_mrrs(arr: np.ndarray) -> list[RoomMRR]:
    return [
        RoomMRR(
            cx=float(r[0]),
            cy=float(r[1]),
            w=max(float(r[2]), 0.0),
            h=max(float(r[3]), 0.0),
            angle=canonical_angle(float(r[4])),
            label_idx=int(round(r[5])),
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
        raise RepairRejected(f"overlap repair too large: {overlap_frac:.3f}")

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
        raise RepairRejected(f"gap repair too large: {gap_frac:.3f}")

    if not gap.is_empty and gap.area > 0:
        for piece in _iter_polygons(gap):
            if piece.area <= 0:
                continue
            idx = _nearest_room_idx(piece, resolved)
            merged = part_union(resolved[idx][0], piece)
            resolved[idx] = (_largest_polygon(merged), resolved[idx][1])

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


def _largest_polygon(geom: BaseGeometry) -> Polygon:
    polys = list(_iter_polygons(geom))
    if not polys:
        return geom if isinstance(geom, Polygon) else Polygon()
    return max(polys, key=lambda p: p.area)


def _nearest_room_idx(piece: Polygon, rooms: list[tuple[Polygon, int]]) -> int:
    best, best_d = 0, float("inf")
    for i, (poly, _) in enumerate(rooms):
        d = poly.distance(piece)
        if d < best_d:
            best, best_d = i, d
    return best
