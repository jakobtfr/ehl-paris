"""Evaluation scoring — aggregate metrics into judge-friendly summary scores.

Produces a single-page scoring summary from raw evaluation results. Designed for
display in README / demo output / CI logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry.base import BaseGeometry

from ..generate import sample_layouts
from .metrics import distribution_metrics, validity_metrics


@dataclass
class ScoreSummary:
    """Aggregated scoring for a batch of generated layouts."""

    n_outlines: int
    n_samples: int
    n_layouts: int
    n_failures: int

    # Validity (lower is better, 0 = perfect)
    outside_frac_mean: float
    overlap_frac_mean: float
    gap_frac_mean: float
    invalid_rate_mean: float
    room_count_mean: float

    # Quality flags
    perfect_partition_rate: float  # fraction with near-zero outside/overlap/gap

    # Distribution
    label_diversity: int  # number of distinct labels used

    @property
    def validity_score(self) -> float:
        """0-100 score: 100 = all layouts are geometrically perfect."""
        penalty = (
            self.outside_frac_mean * 30
            + self.overlap_frac_mean * 30
            + self.gap_frac_mean * 25
            + self.invalid_rate_mean * 15
        )
        return max(0.0, min(100.0, 100.0 - penalty * 100))

    @property
    def overall_score(self) -> float:
        """Weighted overall credibility score (0-100)."""
        success_rate = 1.0 - (self.n_failures / max(self.n_outlines, 1))
        diversity_bonus = min(self.label_diversity / 8.0, 1.0)
        return (
            self.validity_score * 0.6
            + success_rate * 100 * 0.25
            + diversity_bonus * 100 * 0.15
        )

    def to_markdown(self) -> str:
        lines = [
            "## Evaluation Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Outlines evaluated | {self.n_outlines} |",
            f"| Samples per outline | {self.n_samples} |",
            f"| Total layouts | {self.n_layouts} |",
            f"| Failures | {self.n_failures} |",
            f"| Perfect partition rate | {self.perfect_partition_rate:.1%} |",
            f"| Avg room count | {self.room_count_mean:.1f} |",
            f"| Label diversity | {self.label_diversity} types |",
            "",
            "### Validity Breakdown",
            "",
            "| Metric | Mean |",
            "|--------|------|",
            f"| Outside fraction | {self.outside_frac_mean:.4f} |",
            f"| Overlap fraction | {self.overlap_frac_mean:.4f} |",
            f"| Gap fraction | {self.gap_frac_mean:.4f} |",
            f"| Invalid geometry rate | {self.invalid_rate_mean:.4f} |",
            "",
            f"**Validity Score: {self.validity_score:.1f}/100**  ",
            f"**Overall Score: {self.overall_score:.1f}/100**",
        ]
        return "\n".join(lines)


def score_batch(
    outlines: dict[str, BaseGeometry],
    n_samples: int = 4,
    seed: int = 42,
) -> ScoreSummary:
    """Run evaluation and compute aggregated scores."""
    all_validity: list[dict] = []
    all_layouts_tuples: list[list[tuple]] = []
    failures = 0

    for _unit_id, outline in outlines.items():
        try:
            layouts = sample_layouts(outline, seed=seed, n_samples=n_samples)
        except Exception:
            failures += 1
            continue

        for layout in layouts:
            rooms_tuples = [(r["polygon"], r["label_idx"]) for r in layout]
            vm = validity_metrics(rooms_tuples, outline)
            all_validity.append(vm)
            all_layouts_tuples.append(rooms_tuples)

    n_layouts = len(all_validity)

    if n_layouts == 0:
        return ScoreSummary(
            n_outlines=len(outlines), n_samples=n_samples,
            n_layouts=0, n_failures=failures,
            outside_frac_mean=0, overlap_frac_mean=0,
            gap_frac_mean=0, invalid_rate_mean=0,
            room_count_mean=0, perfect_partition_rate=0,
            label_diversity=0,
        )

    dist = distribution_metrics(all_layouts_tuples)
    perfect = sum(
        1 for v in all_validity
        if v["outside_frac"] < 0.01
        and v["overlap_frac"] < 0.01
        and v["gap_frac"] < 0.05
    )

    return ScoreSummary(
        n_outlines=len(outlines),
        n_samples=n_samples,
        n_layouts=n_layouts,
        n_failures=failures,
        outside_frac_mean=float(np.mean([v["outside_frac"] for v in all_validity])),
        overlap_frac_mean=float(np.mean([v["overlap_frac"] for v in all_validity])),
        gap_frac_mean=float(np.mean([v["gap_frac"] for v in all_validity])),
        invalid_rate_mean=float(np.mean([v["invalid_rate"] for v in all_validity])),
        room_count_mean=float(np.mean([v["n_rooms"] for v in all_validity])),
        perfect_partition_rate=perfect / n_layouts,
        label_diversity=len(dist.get("label_freq", {})),
    )
