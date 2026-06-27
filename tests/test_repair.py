"""Focused tests for the MRR deterministic repair layer."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from floorgen.repr.mrr import (
    RepairRejected,
    RoomMRR,
    partition_accounting,
    repair_partition,
)


def test_repair_fills_only_threshold_accepted_gap():
    outline = box(0, 0, 10, 10)
    mrrs = [
        RoomMRR(2.45, 5, 4.9, 10, 0.0, 0),
        RoomMRR(7.55, 5, 4.9, 10, 0.0, 7),
    ]

    repaired = repair_partition(mrrs, outline, max_gap_frac=0.03, min_area=0.01)
    accounting = partition_accounting(repaired, outline)

    assert accounting.gap_frac < 1e-9
    assert accounting.overlap_frac < 1e-9
    assert accounting.outside_frac < 1e-9


def test_repair_rejects_gap_beyond_configured_sliver_budget():
    outline = box(0, 0, 10, 10)
    mrrs = [
        RoomMRR(2, 5, 4, 10, 0.0, 0),
        RoomMRR(8, 5, 4, 10, 0.0, 7),
    ]

    with pytest.raises(RepairRejected, match=r"gap repair too large: 0\.200 > 0\.050"):
        repair_partition(mrrs, outline, max_gap_frac=0.05)


def test_repair_resolves_threshold_accepted_overlap():
    outline = box(0, 0, 10, 10)
    mrrs = [
        RoomMRR(2.55, 5, 5.1, 10, 0.0, 0),
        RoomMRR(7.45, 5, 5.1, 10, 0.0, 7),
    ]

    repaired = repair_partition(mrrs, outline, max_overlap_frac=0.03)
    accounting = partition_accounting(repaired, outline)

    assert accounting.gap_frac < 1e-9
    assert accounting.overlap_frac < 1e-9
    assert accounting.outside_frac < 1e-9


def test_repair_rejects_overlap_beyond_configured_sliver_budget():
    outline = box(0, 0, 10, 10)
    mrrs = [
        RoomMRR(3, 5, 6, 10, 0.0, 0),
        RoomMRR(7, 5, 6, 10, 0.0, 7),
    ]

    with pytest.raises(RepairRejected, match=r"overlap repair too large: 0\.200 > 0\.050"):
        repair_partition(mrrs, outline, max_overlap_frac=0.05)
