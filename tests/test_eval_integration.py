"""Integration tests: end-to-end pipeline from generate through export and evaluation.

These tests verify the full workflow judges would run:
  outline → generate → validate → render → export → report
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from floorgen.config import SEED
from floorgen.eval.metrics import distribution_metrics, validity_metrics
from floorgen.eval.render import RenderConfig, render_batch, render_layout, save_render
from floorgen.export import ExportConfig, export_layouts, export_to_parquet
from floorgen.generate import generate, sample_layouts


@pytest.fixture
def realistic_outline():
    """A more realistic L-shaped apartment outline."""
    from shapely.geometry import Polygon
    coords = [(0, 0), (12, 0), (12, 6), (8, 6), (8, 10), (0, 10), (0, 0)]
    return Polygon(coords)


@pytest.fixture
def rect_outline():
    return box(0, 0, 10, 8)


class TestEndToEndGenerate:
    def test_generate_returns_rooms(self, rect_outline):
        rooms = generate(rect_outline)
        assert isinstance(rooms, list)
        assert len(rooms) > 0
        for r in rooms:
            assert "label" in r
            assert "polygon" in r
            assert "label_idx" in r
            assert "geojson" in r

    def test_sample_layouts_multiple(self, rect_outline):
        layouts = sample_layouts(rect_outline, seed=SEED, n_samples=3)
        assert len(layouts) == 3
        for layout in layouts:
            assert len(layout) > 0

    def test_generate_realistic_outline(self, realistic_outline):
        rooms = generate(realistic_outline)
        assert len(rooms) >= 2


class TestEndToEndValidation:
    def test_generated_layout_validity(self, rect_outline):
        rooms = generate(rect_outline)
        rooms_tuples = [(r["polygon"], r["label_idx"]) for r in rooms]
        vm = validity_metrics(rooms_tuples, rect_outline)
        assert vm["outside_frac"] < 0.05
        assert vm["overlap_frac"] < 0.05
        assert vm["invalid_rate"] == 0.0
        assert vm["n_rooms"] >= 2

    def test_batch_validity_stats(self, rect_outline):
        layouts = sample_layouts(rect_outline, seed=SEED, n_samples=4)
        all_tuples = [
            [(r["polygon"], r["label_idx"]) for r in layout]
            for layout in layouts
        ]
        dist = distribution_metrics(all_tuples)
        assert dist["room_count_mean"] > 0
        assert len(dist["label_freq"]) >= 2


class TestEndToEndRender:
    def test_render_generated_layout(self, rect_outline):
        rooms = generate(rect_outline)
        rooms_tuples = [(r["polygon"], r["label_idx"]) for r in rooms]
        img = render_layout(rooms_tuples, rect_outline)
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8

    def test_render_batch_multiple(self, rect_outline):
        layouts = sample_layouts(rect_outline, seed=SEED, n_samples=3)
        all_tuples = [
            [(r["polygon"], r["label_idx"]) for r in layout]
            for layout in layouts
        ]
        stack = render_batch(all_tuples, rect_outline, cfg=RenderConfig(size=256))
        assert stack.shape == (3, 256, 256, 3)

    def test_save_rendered_image(self, rect_outline, tmp_path):
        rooms = generate(rect_outline)
        rooms_tuples = [(r["polygon"], r["label_idx"]) for r in rooms]
        img = render_layout(rooms_tuples, rect_outline)
        out = tmp_path / "layout.png"
        save_render(img, out)
        assert out.exists()
        assert out.stat().st_size > 500


class TestEndToEndExport:
    def test_export_full_pipeline(self, rect_outline, tmp_path):
        outlines = {"test_001": rect_outline}
        cfg = ExportConfig(
            output_dir=tmp_path / "export",
            n_samples=2,
            seed=SEED,
            checkpoint="test-integration",
            config_notes="integration test",
            include_validity=True,
        )
        path = export_to_parquet(outlines, cfg)
        assert path.exists()

        meta_path = path.with_name(path.stem + "_meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["checkpoint"] == "test-integration"
        assert meta["n_outlines"] == 1
        assert meta["seed"] == SEED

    def test_export_wkt_roundtrips(self, rect_outline):
        from shapely import wkt

        outlines = {"rt_001": rect_outline}
        df = export_layouts(outlines, ExportConfig(n_samples=1, seed=SEED))
        for wkt_str in df["wkt"]:
            geom = wkt.loads(wkt_str)
            assert geom.area > 0

    def test_export_deterministic(self, rect_outline, tmp_path):
        outlines = {"det_001": rect_outline}
        cfg = ExportConfig(output_dir=tmp_path / "det", n_samples=2, seed=SEED)
        df1 = export_layouts(outlines, cfg)
        df2 = export_layouts(outlines, cfg)
        assert df1["wkt"].tolist() == df2["wkt"].tolist()
        assert df1["label"].tolist() == df2["label"].tolist()


class TestEvaluateCLI:
    def test_demo_report_includes_backend_and_renderer(self, tmp_path):
        report_path = tmp_path / "eval.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--demo",
                "--n-samples",
                "1",
                "--output",
                str(report_path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        assert "Layouts generated" in result.stdout
        report = json.loads(report_path.read_text())
        assert report["backend"]["checkpoint"] == "baseline"
        assert report["renderer"]["size"] == 512
