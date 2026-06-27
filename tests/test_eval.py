"""Smoke tests for floorgen.eval — geometry metrics, rendering, PRDC.

These tests are torch-free; they exercise the geometry validity path and the
renderer to ensure the evaluation pipeline works on machines without GPU deps.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from floorgen.config import ROOM_NAMES
from floorgen.eval.metrics import distribution_metrics, validity_metrics
from floorgen.eval.render import ROOM_COLORS, RenderConfig, render_layout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_outline() -> Polygon:
    return box(0, 0, 10, 10)


@pytest.fixture
def simple_rooms(simple_outline: Polygon) -> list[tuple[Polygon, int]]:
    """Two rooms that perfectly partition the outline."""
    return [
        (box(0, 0, 5, 10), 0),  # Bedroom
        (box(5, 0, 10, 10), 2),  # Kitchen
    ]


@pytest.fixture
def overlapping_rooms() -> list[tuple[Polygon, int]]:
    """Two rooms that overlap."""
    return [
        (box(0, 0, 6, 10), 0),
        (box(4, 0, 10, 10), 1),
    ]


# ---------------------------------------------------------------------------
# validity_metrics tests
# ---------------------------------------------------------------------------
class TestValidityMetrics:
    def test_perfect_partition(self, simple_rooms, simple_outline):
        m = validity_metrics(simple_rooms, simple_outline)
        assert m["outside_frac"] == pytest.approx(0.0, abs=1e-9)
        assert m["overlap_frac"] == pytest.approx(0.0, abs=1e-9)
        assert m["gap_frac"] == pytest.approx(0.0, abs=1e-9)
        assert m["invalid_rate"] == 0.0
        assert m["n_rooms"] == 2

    def test_overlap_detected(self, overlapping_rooms, simple_outline):
        m = validity_metrics(overlapping_rooms, simple_outline)
        assert m["overlap_frac"] > 0.0

    def test_rooms_outside(self, simple_outline):
        outside_room = box(10, 10, 15, 15)
        m = validity_metrics([(outside_room, 0)], simple_outline)
        assert m["outside_frac"] > 0.0

    def test_gap_detected(self, simple_outline):
        small_room = box(0, 0, 3, 3)
        m = validity_metrics([(small_room, 0)], simple_outline)
        assert m["gap_frac"] > 0.0

    def test_empty_rooms(self, simple_outline):
        m = validity_metrics([], simple_outline)
        assert m["n_rooms"] == 0
        assert m["gap_frac"] == 1.0

    def test_invalid_geometry(self, simple_outline):
        bowtie = Polygon([(0, 0), (5, 5), (5, 0), (0, 5)])
        assert not bowtie.is_valid
        m = validity_metrics([(bowtie, 0)], simple_outline)
        assert m["invalid_rate"] > 0.0
        assert m["n_rooms"] == 1


# ---------------------------------------------------------------------------
# distribution_metrics tests
# ---------------------------------------------------------------------------
class TestDistributionMetrics:
    def test_basic_stats(self, simple_rooms):
        layouts = [simple_rooms, simple_rooms]
        d = distribution_metrics(layouts)
        assert d["room_count_mean"] == 2.0
        assert d["room_count_std"] == 0.0
        assert "label_freq" in d
        assert sum(d["label_freq"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_layouts(self):
        d = distribution_metrics([])
        assert d["room_count_mean"] == 0.0


# ---------------------------------------------------------------------------
# render_layout tests
# ---------------------------------------------------------------------------
class TestRenderLayout:
    def test_output_shape(self, simple_rooms, simple_outline):
        img = render_layout(simple_rooms, simple_outline)
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8

    def test_custom_size(self, simple_rooms, simple_outline):
        cfg = RenderConfig(size=256)
        img = render_layout(simple_rooms, simple_outline, cfg=cfg)
        assert img.shape == (256, 256, 3)

    def test_non_white_pixels(self, simple_rooms, simple_outline):
        """A layout with rooms should produce coloured pixels."""
        img = render_layout(simple_rooms, simple_outline)
        non_white = np.any(img != 255, axis=-1)
        assert non_white.sum() > 100

    def test_room_colors_mapped(self):
        """Every ROOM_NAME must have a corresponding colour."""
        for name in ROOM_NAMES:
            assert name in ROOM_COLORS
            color = ROOM_COLORS[name]
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


# ---------------------------------------------------------------------------
# PRDC lazy import test
# ---------------------------------------------------------------------------
class TestPRDC:
    def test_compute_prdc_small(self):
        from floorgen.eval.prdc import compute_prdc

        rng = np.random.default_rng(42)
        real = rng.standard_normal((20, 64))
        fake = rng.standard_normal((20, 64))
        result = compute_prdc(real, fake, k=3)
        assert set(result.keys()) == {"precision", "recall", "density", "coverage"}
        for v in result.values():
            assert 0.0 <= v <= 1.5  # density can exceed 1

    def test_prdc_identical(self):
        from floorgen.eval.prdc import compute_prdc

        feats = np.eye(10)
        result = compute_prdc(feats, feats, k=3)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Torch-import guard tests
# ---------------------------------------------------------------------------
class TestLazyImports:
    def test_metrics_module_imports_without_torch(self):
        """The metrics module itself must import cleanly without torch."""
        import importlib
        mod = importlib.import_module("floorgen.eval.metrics")
        assert hasattr(mod, "validity_metrics")
        assert hasattr(mod, "compute_fid")

    def test_prdc_imports_without_torch(self):
        """prdc.py only needs numpy (sklearn is optional)."""
        import importlib
        mod = importlib.import_module("floorgen.eval.prdc")
        assert hasattr(mod, "compute_prdc")


# ---------------------------------------------------------------------------
# render_batch and save_render tests
# ---------------------------------------------------------------------------
class TestRenderBatch:
    def test_batch_shape(self, simple_rooms, simple_outline):
        from floorgen.eval.render import render_batch

        layouts = [simple_rooms, simple_rooms, simple_rooms]
        stack = render_batch(layouts, simple_outline)
        assert stack.shape == (3, 512, 512, 3)
        assert stack.dtype == np.uint8

    def test_batch_custom_size(self, simple_rooms, simple_outline):
        from floorgen.eval.render import render_batch

        cfg = RenderConfig(size=128)
        stack = render_batch([simple_rooms], simple_outline, cfg=cfg)
        assert stack.shape == (1, 128, 128, 3)

    def test_save_render(self, simple_rooms, simple_outline, tmp_path):
        from floorgen.eval.render import save_render

        img = render_layout(simple_rooms, simple_outline)
        out_path = tmp_path / "test_render.png"
        save_render(img, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 100
