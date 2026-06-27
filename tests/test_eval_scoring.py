"""Tests for floorgen.eval.scoring — aggregated evaluation scores."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from floorgen.eval.scoring import ScoreSummary, score_batch


@pytest.fixture
def demo_outlines():
    return {
        "unit_a": box(0, 0, 10, 8),
        "unit_b": box(0, 0, 12, 10),
    }


class TestScoreBatch:
    def test_returns_summary(self, demo_outlines):
        summary = score_batch(demo_outlines, n_samples=2, seed=42)
        assert isinstance(summary, ScoreSummary)
        assert summary.n_outlines == 2
        assert summary.n_samples == 2
        assert summary.n_layouts == 4
        assert summary.n_failures == 0

    def test_validity_score_range(self, demo_outlines):
        summary = score_batch(demo_outlines, n_samples=2, seed=42)
        assert 0 <= summary.validity_score <= 100

    def test_overall_score_range(self, demo_outlines):
        summary = score_batch(demo_outlines, n_samples=2, seed=42)
        assert 0 <= summary.overall_score <= 100

    def test_perfect_partitions(self, demo_outlines):
        summary = score_batch(demo_outlines, n_samples=2, seed=42)
        assert summary.perfect_partition_rate >= 0.0

    def test_label_diversity(self, demo_outlines):
        summary = score_batch(demo_outlines, n_samples=2, seed=42)
        assert summary.label_diversity >= 2

    def test_markdown_output(self, demo_outlines):
        summary = score_batch(demo_outlines, n_samples=2, seed=42)
        md = summary.to_markdown()
        assert "## Evaluation Summary" in md
        assert "Validity Score" in md
        assert "Overall Score" in md
        assert "|" in md

    def test_empty_outlines(self):
        summary = score_batch({}, n_samples=2, seed=42)
        assert summary.n_layouts == 0
        assert summary.n_failures == 0
