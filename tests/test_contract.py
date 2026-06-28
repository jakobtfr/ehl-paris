"""Contract tests for generate(): the guarantees the evaluator relies on."""

from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorgen.baseline import baseline_sample
from floorgen.config import ROOM_NAMES
from floorgen.generate import backend_provenance, generate, sample_layouts
from floorgen.repr.mrr import RepairRejected

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


def test_empty_generator_attempt_is_retried(monkeypatch):
    import floorgen.generate as generate_module

    calls = 0

    def empty_then_baseline(outline, rng):
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return baseline_sample(outline, rng)

    monkeypatch.setattr(generate_module, "GENERATOR", empty_then_baseline)

    layouts = sample_layouts(OUTLINE, seed=42, n_samples=1)

    assert calls == 2
    assert len(layouts[0]) > 0


def test_empty_generator_rejects_after_retries(monkeypatch):
    import floorgen.generate as generate_module

    monkeypatch.setattr(generate_module, "GENERATOR", lambda _outline, _rng: [])

    try:
        sample_layouts(OUTLINE, seed=42, n_samples=1)
    except RepairRejected as exc:
        assert "no repairable rooms" in str(exc)
    else:
        raise AssertionError("empty generator should be rejected")


def test_backend_provenance_reports_checkpoint_env(monkeypatch):
    monkeypatch.setenv("FLOORGEN_CHECKPOINT", "checkpoints/flow.pt")
    monkeypatch.setenv("FLOORGEN_DEVICE", "cpu")
    monkeypatch.setenv("FLOORGEN_SAMPLE_STEPS", "64")
    monkeypatch.setenv("FLOORGEN_PRESENCE_THRESHOLD", "0.4")

    provenance = backend_provenance()

    assert provenance["backend"] == "flow-checkpoint"
    assert provenance["checkpoint"] == "checkpoints/flow.pt"
    assert provenance["steps"] == "64"
