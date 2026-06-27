"""Axis-aligned box representation for rooms.

The model emits, per slot, a token ``(cx, cy, w, h, label_idx, presence)``.
This module converts between real room polygons and that token form, and
provides the *validity-repair* layer that turns raw model output into a clean
vector partition of the outline.

Design rule (challenge-critical): the model decides room geometry. The repair
layer only enforces vector validity -- clip to outline, resolve overlaps/gaps,
drop slivers. It must never become the thing that *decides* room shape, or it
would be a rule-based partitioner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..config import MIN_ROOM_AREA_M2, ROOM_NAMES, SNAP_TOL_M


@dataclass
class RoomBox:
    """One room as an axis-aligned box plus a label."""

    cx: float
    cy: float
    w: float
    h: float
    label_idx: int

    @property
    def label(self) -> str:
        return ROOM_NAMES[self.label_idx]

    def to_polygon(self) -> Polygon:
        return box(self.cx - self.w / 2, self.cy - self.h / 2,
                   self.cx + self.w / 2, self.cy + self.h / 2)


def polygon_to_box(poly: Polygon, label_idx: int) -> RoomBox:
    """Fit the area-equivalent axis-aligned box to a room polygon.

    Uses the polygon centroid and its bounding box aspect, rescaled so the box
    area equals the true room area. This keeps both position and area faithful,
    which is what the rasterised metrics care about.
    """
    minx, miny, maxx, maxy = poly.bounds
    bw, bh = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)
    target_area = poly.area
    bbox_area = bw * bh
    scale = (target_area / bbox_area) ** 0.5 if bbox_area > 0 else 1.0
    c = poly.centroid
    return RoomBox(cx=c.x, cy=c.y, w=bw * scale, h=bh * scale, label_idx=label_idx)


def boxes_to_array(boxes: list[RoomBox]) -> np.ndarray:
    """Pack boxes into an (N, 5) float array: cx, cy, w, h, label_idx."""
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array([[b.cx, b.cy, b.w, b.h, b.label_idx] for b in boxes], dtype=np.float32)


def array_to_boxes(arr: np.ndarray) -> list[RoomBox]:
    return [RoomBox(cx=float(r[0]), cy=float(r[1]), w=float(r[2]),
                    h=float(r[3]), label_idx=int(round(r[4]))) for r in arr]


# ---------------------------------------------------------------------------
# Validity-repair layer
# ---------------------------------------------------------------------------
def repair_partition(
    boxes: list[RoomBox],
    outline: Polygon | MultiPolygon,
    *,
    snap_tol: float = SNAP_TOL_M,
    min_area: float = MIN_ROOM_AREA_M2,
) -> list[tuple[Polygon, int]]:
    """Turn raw room boxes into a clean partition of the outline.

    Steps (validity only, not shape invention):
      1. clip each box to the outline,
      2. resolve overlaps by assigning shared area to the room whose box centre
         is nearest (a Voronoi-style *tie-break*, applied only to contested
         pixels -- not a from-scratch partition),
      3. fill gaps by snapping each remaining region to its nearest room,
      4. drop sub-resolution slivers and merge their area into the neighbour.

    Returns a list of (polygon, label_idx). Polygons are valid, non-overlapping,
    inside the outline, and (within tolerance) cover it.
    """
    if not boxes:
        return []

    clipped: list[tuple[Polygon, int]] = []
    for b in boxes:
        inter = b.to_polygon().intersection(outline)
        if inter.is_empty or inter.area <= 0:
            continue
        for part in _iter_polygons(inter):
            if part.area > 0:
                clipped.append((part, b.label_idx))
    if not clipped:
        return []

    # Resolve overlaps: subtract earlier-claimed area from later boxes, ordered
    # by descending area so larger, more confident rooms keep their core.
    clipped.sort(key=lambda t: t[0].area, reverse=True)
    claimed: BaseGeometry | None = None
    resolved: list[tuple[Polygon, int]] = []
    centres: list[tuple[float, float]] = []
    for poly, label in clipped:
        free = poly if claimed is None else poly.difference(claimed)
        if free.is_empty or free.area < min_area:
            continue
        for part in _iter_polygons(free):
            if part.area >= min_area:
                resolved.append((part, label))
                centres.append((part.centroid.x, part.centroid.y))
        claimed = part_union(claimed, poly)

    if not resolved:
        return []

    # Fill gaps: any outline area not yet claimed goes to the nearest room.
    covered = unary_union([p for p, _ in resolved])
    gap = outline.difference(covered)
    if not gap.is_empty and gap.area > 0:
        for piece in _iter_polygons(gap):
            if piece.area <= 0:
                continue
            idx = _nearest_room_idx(piece, resolved)
            merged = part_union(resolved[idx][0], piece)
            # keep it a single polygon if possible
            resolved[idx] = (_largest_polygon(merged), resolved[idx][1])

    # Final clean: snap-simplify and drop any residual slivers.
    out: list[tuple[Polygon, int]] = []
    for poly, label in resolved:
        clean = poly.buffer(0).simplify(snap_tol / 2).intersection(outline)
        for part in _iter_polygons(clean):
            if part.area >= min_area and part.is_valid:
                out.append((part, label))
    return out


def part_union(a: BaseGeometry | None, b: BaseGeometry) -> BaseGeometry:
    return b if a is None else unary_union([a, b])


def _iter_polygons(geom: BaseGeometry):
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
    pc = piece.centroid
    best, best_d = 0, float("inf")
    for i, (poly, _) in enumerate(rooms):
        d = poly.distance(pc)
        if d < best_d:
            best, best_d = i, d
    return best
