"""Box representation round-trip + repair validity tests."""

from __future__ import annotations

from shapely.geometry import box

from floorgen.repr.boxes import (
    RoomBox,
    array_to_boxes,
    boxes_to_array,
    polygon_to_box,
    repair_partition,
)


def test_polygon_to_box_preserves_area():
    p = box(0, 0, 4, 2)  # area 8
    b = polygon_to_box(p, label_idx=0)
    assert abs(b.w * b.h - p.area) < 1e-6
    assert abs(b.cx - 2) < 1e-6 and abs(b.cy - 1) < 1e-6


def test_array_roundtrip():
    boxes = [RoomBox(1, 1, 2, 2, 0), RoomBox(5, 5, 3, 1, 4)]
    arr = boxes_to_array(boxes)
    assert arr.shape == (2, 5)
    back = array_to_boxes(arr)
    for a, b in zip(boxes, back):
        assert (a.cx, a.cy, a.w, a.h, a.label_idx) == (b.cx, b.cy, b.w, b.h, b.label_idx)


def test_repair_contains_and_partitions():
    outline = box(0, 0, 10, 10)
    boxes = [RoomBox(2.5, 5, 6, 12, 0), RoomBox(7.5, 5, 6, 12, 7)]  # overlapping, overflowing
    part = repair_partition(boxes, outline)
    assert len(part) >= 1
    total = sum(p.area for p, _ in part)
    # contained
    for p, _ in part:
        assert p.difference(outline.buffer(1e-6)).area < 1e-6
    # near-complete coverage, no large overlap
    assert total <= outline.area * 1.001
    assert total >= outline.area * 0.8
