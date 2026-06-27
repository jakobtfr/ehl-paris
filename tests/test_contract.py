"""Contract tests for generate(): the guarantees the evaluator relies on."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorgen.config import ROOM_NAMES
from floorgen.generate import generate, sample_layouts

OUTLINE = Polygon([(0, 0), (10, 0), (10, 6), (6, 6), (6, 10), (0, 10)])
TOL = 0.02


def _polys(layout):
    return [r["polygon"] for r in layout]


def test_returns_room_records():
    layout = generate(OUTLINE)
    assert len(layout) >= 1
    for r in layout:
        assert set(r) >= {"label", "label_idx", "polygon", "geojson"}
        assert r["label"] in ROOM_NAMES


def test_all_polygons_valid():
    assert all(r["polygon"].is_valid for r in generate(OUTLINE))


def test_rooms_inside_outline():
    for p in _polys(generate(OUTLINE)):
        assert p.difference(OUTLINE.buffer(1e-6)).area / OUTLINE.area < TOL


def test_no_pairwise_overlap():
    polys = _polys(generate(OUTLINE))
    total = sum(p.area for p in polys)
    union = unary_union(polys).area
    assert (total - union) / OUTLINE.area < TOL


def test_partitions_outline():
    polys = _polys(generate(OUTLINE))
    covered = unary_union(polys).intersection(OUTLINE).area
    assert covered / OUTLINE.area > 1 - 0.15


def test_determinism_single_sample():
    a, b = generate(OUTLINE), generate(OUTLINE)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra["label_idx"] == rb["label_idx"]
        assert ra["polygon"].symmetric_difference(rb["polygon"]).area < 1e-9


def test_multisample_distinct_but_deterministic():
    s1 = sample_layouts(OUTLINE, seed=42, n_samples=3)
    s2 = sample_layouts(OUTLINE, seed=42, n_samples=3)
    assert len(s1) == 3 and len(s1[0]) == len(s2[0])
    labels = [tuple(r["label_idx"] for r in s) for s in s1]
    sizes = [len(s) for s in s1]
    assert len(set(labels)) > 1 or len(set(sizes)) > 1


def test_geojson_roundtrips():
    import shapely.geometry as sg
    for r in generate(OUTLINE):
        g = sg.shape(r["geojson"])
        assert g.is_valid and abs(g.area - r["polygon"].area) < 1e-6
