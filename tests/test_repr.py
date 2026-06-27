"""MRR representation round-trip + repair validity tests."""

from __future__ import annotations

import math

from shapely import affinity
from shapely.geometry import MultiPolygon, box

from floorgen.repr.mrr import (
    RoomMRR,
    array_to_mrrs,
    mrrs_to_array,
    polygon_to_mrr,
    repair_partition,
    wrapped_angle_distance,
)


def test_polygon_to_mrr_preserves_rotated_rectangle_geometry():
    p = affinity.rotate(box(0, 0, 4, 2), 30, origin="centroid")
    mrr = polygon_to_mrr(p, label_idx=0)
    decoded = mrr.to_polygon()
    iou = decoded.intersection(p).area / decoded.union(p).area
    assert iou > 0.999
    assert mrr.w >= mrr.h


def test_angle_distance_wraps_modulo_pi():
    assert wrapped_angle_distance(0.0, math.pi) < 1e-9
    assert wrapped_angle_distance(math.pi / 2 - 0.01, -math.pi / 2 + 0.01) < 0.03


def test_room_mrr_canonicalizes_dimensions_without_changing_geometry():
    raw = affinity.translate(
        affinity.rotate(box(-1, -3, 1, 3), 0.35, origin=(0, 0), use_radians=True),
        xoff=5,
        yoff=-2,
    )

    mrr = RoomMRR(5, -2, 2, 6, 0.35, 1)

    assert mrr.w == 6
    assert mrr.h == 2
    assert wrapped_angle_distance(mrr.angle, 0.35 + math.pi / 2) < 1e-9
    assert mrr.to_polygon().symmetric_difference(raw).area < 1e-9


def test_array_roundtrip():
    mrrs = [RoomMRR(1, 1, 2, 2, 0.2, 0), RoomMRR(5, 5, 3, 1, -0.4, 4)]
    arr = mrrs_to_array(mrrs)
    assert arr.shape == (2, 6)
    back = array_to_mrrs(arr)
    for a, b in zip(mrrs, back):
        assert (a.cx, a.cy, a.w, a.h, a.label_idx) == (b.cx, b.cy, b.w, b.h, b.label_idx)
        assert wrapped_angle_distance(a.angle, b.angle) < 1e-6


def test_repair_contains_and_partitions():
    outline = box(0, 0, 10, 10)
    mrrs = [
        RoomMRR(2.5, 5, 6, 12, 0.0, 0),
        RoomMRR(7.5, 5, 6, 12, 0.0, 7),
    ]
    part = repair_partition(mrrs, outline)
    assert len(part) >= 1
    total = sum(p.area for p, _ in part)
    for p, _ in part:
        assert p.difference(outline.buffer(1e-6)).area < 1e-6
    assert total <= outline.area * 1.001
    assert total >= outline.area * 0.8


def test_repair_keeps_accepted_disconnected_gap_piece():
    outline = MultiPolygon([box(0, 0, 2, 2), box(3, 0, 4, 1)])
    part = repair_partition(
        [RoomMRR(1, 1, 2, 2, 0.0, 0)],
        outline,
        min_area=0.01,
        max_gap_frac=0.25,
    )

    assert len(part) == 2
    assert sum(p.area for p, _ in part) == outline.area
    assert all(label == 0 for _, label in part)
    for p, _ in part:
        assert p.difference(outline).area < 1e-9
