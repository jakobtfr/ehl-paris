"""Room-count distribution report.

MAX_ROOMS_K (the fixed slot count for the generative model) is meant to be read
off the processed room-count distribution. ``room_count_distribution`` exposes a
histogram, percentiles, and a suggested slot count so that choice is data-driven
and auditable rather than guessed. Synthetic records only.
"""

from __future__ import annotations

from floorgen.data.preprocess import room_count_distribution


def _recs(counts: list[int]) -> list[dict]:
    return [{"n_rooms": c} for c in counts]


def test_uniform_counts_collapse_to_single_value():
    dist = room_count_distribution(_recs([5, 5, 5, 5]))
    assert dist["histogram"] == {5: 4}
    assert dist["p50"] == 5
    assert dist["p99"] == 5
    assert dist["suggested_max_rooms_k"] == 5


def test_histogram_counts_each_room_count():
    dist = room_count_distribution(_recs([2, 2, 3, 5, 9]))
    assert dist["histogram"] == {2: 2, 3: 1, 5: 1, 9: 1}


def test_percentiles_are_monotonic_non_decreasing():
    dist = room_count_distribution(_recs([2, 2, 3, 5, 9, 12, 4, 6]))
    assert dist["p50"] <= dist["p90"] <= dist["p95"] <= dist["p99"]


def test_suggested_slot_count_covers_the_99th_percentile():
    counts = [2, 2, 3, 5, 9, 12, 4, 6]
    dist = room_count_distribution(counts and _recs(counts))
    assert dist["suggested_max_rooms_k"] >= dist["p99"]
    assert dist["suggested_max_rooms_k"] <= max(counts)


def test_empty_records_are_safe():
    dist = room_count_distribution([])
    assert dist["histogram"] == {}
    assert dist["suggested_max_rooms_k"] == 0
