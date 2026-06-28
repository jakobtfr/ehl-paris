"""Ranked post-processing for generated room MRR candidates.

Raw generation remains the honest model output plus strict deterministic repair.
Ranked mode spends extra test-time compute on multiple raw candidates, scores
them for geometric and layout plausibility, and selects a diverse subset. When
strict repair rejects a candidate, ranked mode may use permissive repair, but it
penalizes that candidate so the provenance does not blur model quality with
repair quality.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .config import MAX_ROOMS_K, ROOM_NAME_TO_IDX, ROOM_NAMES
from .eval.metrics import validity_metrics
from .repr.mrr import RepairRejected, RoomMRR, repair_partition

GeneratorFn = Callable[[BaseGeometry, np.random.Generator], list[RoomMRR]]
Partition = list[tuple[Polygon, int]]


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for inspectable ranked-mode candidate scoring."""

    validity: float = 100.0
    raw_repair_pressure: float = 18.0
    room_count: float = 4.0
    area_ratio: float = 8.0
    label_mix: float = 6.0
    permissive_repair: float = 10.0


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for ranked candidate generation and selection."""

    candidate_budget: int = 16
    max_outside_frac: float = 0.02
    max_overlap_frac: float = 0.02
    max_gap_frac: float = 0.08
    weights: ScoringWeights = ScoringWeights()


@dataclass(frozen=True)
class RankedCandidate:
    """One generated candidate after repair and scoring."""

    index: int
    partition: Partition
    score: float
    signature: str
    repair_mode: str
    raw_metrics: dict[str, float]
    repaired_metrics: dict[str, float]
    rejection_reason: str | None = None
    semantic_repair: dict[str, Any] | None = None

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


@dataclass(frozen=True)
class RankedSelection:
    """Selected layouts plus JSON-serializable ranking provenance."""

    layouts: list[Partition]
    candidates: list[RankedCandidate]
    provenance: dict[str, Any]


def plausible_room_count_range(outline: BaseGeometry) -> tuple[int, int, int]:
    """Area-conditioned room-count prior for Swiss apartment-scale outlines."""

    area = max(float(outline.area), 0.0)
    expected = int(np.clip(round(3.0 + area / 14.0), 3, MAX_ROOMS_K))
    slack = max(2, int(round(expected * 0.35)))
    low = max(1, expected - slack)
    high = min(MAX_ROOMS_K, expected + slack)
    if area >= 90.0:
        high = min(MAX_ROOMS_K, high + 2)
    if area <= 35.0:
        high = min(high, 8)
    return low, expected, high


def layout_signature(partition: Partition, outline: BaseGeometry) -> str:
    """Stable coarse signature used for same-outline diversity selection."""

    outline_area = max(float(outline.area), 1e-9)
    pieces = []
    for poly, label_idx in sorted(partition, key=lambda item: (item[1], -item[0].area)):
        point = poly.representative_point()
        pieces.append(
            f"{int(label_idx)}:{poly.area / outline_area:.3f}:"
            f"{point.x:.1f}:{point.y:.1f}"
        )
    return hashlib.sha1("|".join(pieces).encode("utf-8")).hexdigest()[:12]


def rank_samples(
    outline: BaseGeometry,
    generator: GeneratorFn,
    rng: np.random.Generator,
    *,
    n_samples: int = 1,
    config: RankingConfig | None = None,
) -> RankedSelection:
    """Generate a candidate pool, rank it, and return diverse repaired layouts."""

    config = config or RankingConfig()
    n_samples = max(1, int(n_samples))
    candidate_budget = max(n_samples, int(config.candidate_budget))

    candidates = [
        _candidate_from_mrrs(generator(outline, rng), outline, idx, config)
        for idx in range(candidate_budget)
    ]
    accepted = sorted(
        (candidate for candidate in candidates if candidate.accepted),
        key=lambda candidate: (candidate.score, candidate.signature),
    )
    if not accepted:
        first_reason = next(
            (candidate.rejection_reason for candidate in candidates if candidate.rejection_reason),
            "no candidates generated",
        )
        raise RepairRejected(f"ranked post-processing found no acceptable candidates: {first_reason}")

    selected = _select_diverse(accepted, n_samples)
    provenance = _provenance(config, candidates, selected)
    return RankedSelection(
        layouts=[candidate.partition for candidate in selected],
        candidates=candidates,
        provenance=provenance,
    )


def rank_layouts(
    outline: BaseGeometry,
    generator: GeneratorFn,
    rng: np.random.Generator,
    *,
    n_samples: int = 1,
    config: RankingConfig | None = None,
) -> RankedSelection:
    """Compatibility alias for demo/status discovery."""

    return rank_samples(
        outline,
        generator,
        rng,
        n_samples=n_samples,
        config=config,
    )


def _candidate_from_mrrs(
    mrrs: list[RoomMRR],
    outline: BaseGeometry,
    index: int,
    config: RankingConfig,
) -> RankedCandidate:
    raw_parts = [
        (mrr.to_polygon(), mrr.label_idx)
        for mrr in mrrs
        if mrr.has_finite_geometry and mrr.w > 0.0 and mrr.h > 0.0
    ]
    raw_metrics = validity_metrics(raw_parts, outline)

    if not raw_parts:
        return _rejected_candidate(index, "empty raw MRR set", raw_metrics)

    repair_mode = "strict"
    strict_error = None
    try:
        partition = repair_partition(mrrs, outline)
    except RepairRejected as exc:
        strict_error = str(exc)
        repair_mode = "permissive"
        partition = repair_partition(mrrs, outline, reject_large_repairs=False)

    if not partition:
        detail = f"strict repair rejected: {strict_error}" if strict_error else "empty partition"
        return _rejected_candidate(index, detail, raw_metrics)

    partition, semantic_repair = _calibrate_collapsed_labels(partition, outline)
    repaired_metrics = validity_metrics(partition, outline)
    rejection_reason = _hard_rejection_reason(repaired_metrics, config)
    score = _score_candidate(
        partition,
        outline,
        raw_metrics,
        repaired_metrics,
        repair_mode,
        config.weights,
    )
    if strict_error:
        repaired_metrics = {**repaired_metrics, "strict_repair_rejected": 1.0}
    return RankedCandidate(
        index=index,
        partition=partition,
        score=score,
        signature=layout_signature(partition, outline),
        repair_mode=repair_mode,
        raw_metrics=raw_metrics,
        repaired_metrics=repaired_metrics,
        rejection_reason=rejection_reason,
        semantic_repair=semantic_repair,
    )


def _rejected_candidate(
    index: int,
    reason: str,
    raw_metrics: dict[str, float] | None = None,
) -> RankedCandidate:
    metrics = raw_metrics or {
        "outside_frac": 0.0,
        "overlap_frac": 0.0,
        "gap_frac": 1.0,
        "invalid_rate": 0.0,
        "n_rooms": 0,
    }
    return RankedCandidate(
        index=index,
        partition=[],
        score=float("inf"),
        signature=f"rejected-{index}",
        repair_mode="none",
        raw_metrics=metrics,
        repaired_metrics=metrics,
        rejection_reason=reason,
        semantic_repair=None,
    )


def _hard_rejection_reason(metrics: dict[str, float], config: RankingConfig) -> str | None:
    if metrics["n_rooms"] <= 0:
        return "empty repaired partition"
    if metrics["invalid_rate"] > 0.0:
        return f"invalid polygon rate {metrics['invalid_rate']:.3f}"
    if metrics["outside_frac"] > config.max_outside_frac:
        return f"outside fraction {metrics['outside_frac']:.3f}"
    if metrics["overlap_frac"] > config.max_overlap_frac:
        return f"overlap fraction {metrics['overlap_frac']:.3f}"
    if metrics["gap_frac"] > config.max_gap_frac:
        return f"gap fraction {metrics['gap_frac']:.3f}"
    return None


def _score_candidate(
    partition: Partition,
    outline: BaseGeometry,
    raw_metrics: dict[str, float],
    repaired_metrics: dict[str, float],
    repair_mode: str,
    weights: ScoringWeights,
) -> float:
    validity = (
        repaired_metrics["outside_frac"]
        + repaired_metrics["overlap_frac"]
        + repaired_metrics["gap_frac"]
        + repaired_metrics["invalid_rate"]
    )
    raw_pressure = (
        raw_metrics["outside_frac"]
        + raw_metrics["overlap_frac"]
        + 0.5 * raw_metrics["gap_frac"]
    )
    score = weights.validity * validity
    score += weights.raw_repair_pressure * raw_pressure
    score += weights.room_count * _room_count_penalty(len(partition), outline)
    score += weights.area_ratio * _area_ratio_penalty(partition, outline)
    score += weights.label_mix * _label_mix_penalty(partition, outline)
    if repair_mode == "permissive":
        score += weights.permissive_repair
    return float(score)


def _room_count_penalty(n_rooms: int, outline: BaseGeometry) -> float:
    low, expected, high = plausible_room_count_range(outline)
    if n_rooms < low:
        return float(low - n_rooms + abs(n_rooms - expected) * 0.2)
    if n_rooms > high:
        return float(n_rooms - high + abs(n_rooms - expected) * 0.2)
    return float(abs(n_rooms - expected) * 0.1)


def _area_ratio_penalty(partition: Partition, outline: BaseGeometry) -> float:
    outline_area = max(float(outline.area), 1e-9)
    ratios = [float(poly.area) / outline_area for poly, _ in partition if poly.area > 0]
    if not ratios:
        return 10.0
    penalty = 0.0
    for ratio in ratios:
        if ratio < 0.006:
            penalty += (0.006 - ratio) * 100.0
        if ratio > 0.55:
            penalty += (ratio - 0.55) * 10.0
    if max(ratios) > 0.72 and len(ratios) > 1:
        penalty += 2.0
    return float(penalty)


def _label_mix_penalty(partition: Partition, outline: BaseGeometry) -> float:
    area = max(float(outline.area), 0.0)
    labels = {ROOM_NAMES[label_idx] for _, label_idx in partition}
    expected = {"Bathroom", "Kitchen"}
    if area >= 45.0:
        expected.add("Livingroom")
    if area >= 55.0:
        expected.add("Bedroom")

    penalty = float(len(expected - labels))
    if len(partition) >= 6 and len(labels) < 3:
        penalty += 1.5
    return penalty


def _calibrate_collapsed_labels(
    partition: Partition,
    outline: BaseGeometry,
) -> tuple[Partition, dict[str, Any] | None]:
    """Apply an explicit semantic fallback when a checkpoint collapses labels.

    Geometry still comes from the model and repair layer. This only handles the
    observed failure mode where a poor type head assigns almost every repaired
    room to one class, which would make rendered FID meaningless.
    """

    n_rooms = len(partition)
    if n_rooms < 4:
        return partition, None

    counts = Counter(label for _, label in partition)
    dominant_label, dominant_count = counts.most_common(1)[0]
    dominant_frac = dominant_count / n_rooms
    if dominant_frac < 0.85 and not (n_rooms >= 6 and len(counts) < 3):
        return partition, None

    label_sequence = _semantic_label_sequence(n_rooms, outline)
    order = sorted(range(n_rooms), key=lambda idx: partition[idx][0].area, reverse=True)
    assigned = [label for _, label in partition]
    for rank, idx in enumerate(order):
        assigned[idx] = label_sequence[rank]

    calibrated = [
        (poly, int(assigned[idx]))
        for idx, (poly, _label) in enumerate(partition)
    ]
    after_counts = Counter(label for _, label in calibrated)
    return calibrated, {
        "applied": True,
        "strategy": "area_ordered_msd_semantic_prior",
        "reason": (
            f"dominant label {ROOM_NAMES[dominant_label]} covered "
            f"{dominant_count}/{n_rooms} rooms"
        ),
        "before_label_counts": {
            ROOM_NAMES[label]: int(count)
            for label, count in counts.most_common()
        },
        "after_label_counts": {
            ROOM_NAMES[label]: int(count)
            for label, count in after_counts.most_common()
        },
    }


def _semantic_label_sequence(n_rooms: int, outline: BaseGeometry) -> list[int]:
    """Deterministic room-label prior used only for collapsed ranked candidates."""

    if n_rooms <= 0:
        return []
    area = max(float(outline.area), 0.0)
    essentials = ["Livingroom", "Bedroom", "Kitchen", "Bathroom"]
    if area < 35.0:
        essentials = ["Bedroom", "Kitchen", "Bathroom", "Livingroom"]
    if area >= 75.0:
        essentials = ["Livingroom", "Bedroom", "Bedroom", "Kitchen", "Bathroom"]

    fill = [
        "Corridor",
        "Bedroom",
        "Balcony",
        "Structure",
        "Bathroom",
        "Kitchen",
        "Storeroom",
        "Bedroom",
        "Structure",
        "Balcony",
        "Livingroom",
        "Dining",
        "Stairs",
    ]
    names: list[str] = []
    for name in essentials:
        if len(names) < n_rooms:
            names.append(name)
    i = 0
    while len(names) < n_rooms:
        names.append(fill[i % len(fill)])
        i += 1
    return [ROOM_NAME_TO_IDX[name] for name in names]


def _select_diverse(candidates: list[RankedCandidate], n_samples: int) -> list[RankedCandidate]:
    selected: list[RankedCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.signature in seen:
            continue
        selected.append(candidate)
        seen.add(candidate.signature)
        if len(selected) == n_samples:
            return selected

    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) == n_samples:
            return selected
    return selected


def _provenance(
    config: RankingConfig,
    candidates: list[RankedCandidate],
    selected: list[RankedCandidate],
) -> dict[str, Any]:
    accepted = [candidate for candidate in candidates if candidate.accepted]
    semantic_repairs = [
        candidate for candidate in candidates
        if candidate.semantic_repair and candidate.semantic_repair.get("applied")
    ]
    return {
        "mode": "ranked",
        "candidate_budget": int(config.candidate_budget),
        "scoring_weights": asdict(config.weights),
        "generated_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(candidates) - len(accepted),
        "semantic_repair_count": len(semantic_repairs),
        "selected_indices": [candidate.index for candidate in selected],
        "selected_signatures": [candidate.signature for candidate in selected],
        "selected_semantic_repairs": [
            candidate.semantic_repair
            for candidate in selected
            if candidate.semantic_repair and candidate.semantic_repair.get("applied")
        ],
        "candidates": [
            {
                "index": candidate.index,
                "accepted": candidate.accepted,
                "score": None if not np.isfinite(candidate.score) else round(candidate.score, 6),
                "signature": candidate.signature,
                "repair_mode": candidate.repair_mode,
                "raw_metrics": _rounded_metrics(candidate.raw_metrics),
                "repaired_metrics": _rounded_metrics(candidate.repaired_metrics),
                "rejection_reason": candidate.rejection_reason,
                "semantic_repair": candidate.semantic_repair,
            }
            for candidate in candidates
        ],
    }


def _rounded_metrics(metrics: dict[str, float]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key == "n_rooms":
            out[key] = int(value)
        else:
            out[key] = round(float(value), 6)
    return out
