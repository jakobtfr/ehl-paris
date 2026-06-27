"""Outline construction + shell selection.

``build_outline`` is the single source of truth for the apartment conditioning
boundary: buffer each room out by the wall-bridge distance, union, buffer back
in. ``largest_shell`` reduces a disconnected outline to its biggest piece for
APIs that need one connected shell. Synthetic geometry only.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from floorgen.data.outline import build_outline, largest_shell


def _sq(x0: float, y0: float, s: float = 3.0) -> Polygon:
    return Polygon([(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)])


# --- build_outline ---------------------------------------------------------

def test_empty_room_list_raises():
    with pytest.raises(ValueError, match="zero rooms"):
        build_outline([])


def test_single_room_returns_its_polygon():
    # Buffer-out then buffer-in rounds corners slightly, so allow ~1% slack.
    out = build_outline([_sq(0, 0)])
    assert isinstance(out, Polygon)
    assert math.isclose(out.area, 9.0, rel_tol=1e-2)


def test_overlapping_rooms_fuse_without_double_counting_area():
    # A(0,0)-(3,3) and B(2,0)-(5,3) union to the clean rectangle (0,0)-(5,3).
    out = build_outline([_sq(0, 0), _sq(2, 0)])
    assert isinstance(out, Polygon)
    assert math.isclose(out.area, 15.0, rel_tol=1e-2)


def test_edge_sharing_rooms_become_one_polygon():
    out = build_outline([_sq(0, 0), _sq(3, 0)])
    assert isinstance(out, Polygon)
    assert math.isclose(out.area, 18.0, rel_tol=1e-2)


def test_wall_gap_within_bridge_distance_is_fused():
    # A 0.2 m gap (< 2 * 0.3 m bridge) is closed into a single shell.
    out = build_outline([_sq(0, 0), _sq(3.2, 0)])
    assert isinstance(out, Polygon)


def test_far_apart_rooms_stay_disconnected():
    out = build_outline([_sq(0, 0), _sq(10, 0)])
    assert isinstance(out, MultiPolygon)
    assert len(out.geoms) == 2


# --- largest_shell ---------------------------------------------------------

def test_largest_shell_passes_through_polygon():
    poly = _sq(0, 0)
    assert largest_shell(poly) is poly


def test_largest_shell_picks_biggest_piece():
    small = _sq(0, 0, s=2.0)      # area 4
    big = _sq(100, 0, s=5.0)      # area 25
    shell = largest_shell(MultiPolygon([small, big]))
    assert math.isclose(shell.area, 25.0, rel_tol=1e-6)


def test_largest_shell_preserves_holes_in_chosen_piece():
    holed = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        holes=[[(3, 3), (3, 5), (5, 5), (5, 3)]],
    )
    small = _sq(100, 0, s=1.0)
    shell = largest_shell(MultiPolygon([holed, small]))
    assert len(shell.interiors) == 1


def test_largest_shell_rejects_non_polygonal_geometry():
    with pytest.raises(TypeError):
        largest_shell(Point(0, 0))
