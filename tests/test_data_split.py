"""Train/val split determinism, leakage-safety, and the split summary report.

The split groups by ``plan_id`` so apartments from the same physical floor never
straddle the train/val boundary, and it must be reproducible from the seed alone.
``split_summary`` surfaces unit/plan counts and an explicit leakage check for the
preprocessing report.

Uses minimal synthetic records (unit_id + plan_id) — no geometry or MSD data.
"""

from __future__ import annotations

import random

from floorgen.data.preprocess import split_by_plan, split_summary


def _recs(pairs: list[tuple[int, int]]) -> list[dict]:
    """pairs of (unit_id, plan_id) -> minimal records."""
    return [{"unit_id": u, "plan_id": p} for u, p in pairs]


# Ten plans, three units each (units 0..29, plan = unit // 3).
MANY = _recs([(u, u // 3) for u in range(30)])


# --- split_by_plan ---------------------------------------------------------

def test_split_is_deterministic_for_same_seed():
    assert split_by_plan(MANY, seed=42) == split_by_plan(MANY, seed=42)


def test_split_independent_of_record_order():
    shuffled = MANY[:]
    random.Random(0).shuffle(shuffled)
    assert split_by_plan(shuffled, seed=42) == split_by_plan(MANY, seed=42)


def test_split_is_leakage_safe_every_plan_single_split():
    split = split_by_plan(MANY, seed=42)
    by_plan: dict[int, set[str]] = {}
    for r in MANY:
        by_plan.setdefault(r["plan_id"], set()).add(split[r["unit_id"]])
    assert all(len(s) == 1 for s in by_plan.values())


def test_val_fraction_is_applied_at_plan_level():
    split = split_by_plan(MANY, val_frac=0.2, seed=42)  # int(10 * 0.2) = 2 val plans
    val_plans = {r["plan_id"] for r in MANY if split[r["unit_id"]] == "val"}
    assert len(val_plans) == 2


def test_at_least_one_val_plan_even_for_tiny_input():
    tiny = _recs([(0, 0), (1, 1)])  # 2 plans, val_frac default 0.15 -> int = 0
    split = split_by_plan(tiny, seed=42)
    assert "val" in set(split.values())


# --- split_summary ---------------------------------------------------------

def test_split_summary_counts_units_and_plans():
    split = split_by_plan(MANY, val_frac=0.2, seed=42)
    summary = split_summary(MANY, split)
    assert summary["unit_counts"]["train"] + summary["unit_counts"]["val"] == 30
    assert summary["plan_counts"]["train"] + summary["plan_counts"]["val"] == 10
    assert summary["n_plans"] == 10
    assert summary["plan_leakage"] == []


def test_split_summary_detects_leakage():
    recs = _recs([(0, 5), (1, 5)])  # same plan 5, two units
    leaky = {0: "train", 1: "val"}  # deliberately straddles
    summary = split_summary(recs, leaky)
    assert summary["plan_leakage"] == [5]
