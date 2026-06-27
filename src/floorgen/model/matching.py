"""Hungarian matching utilities for unordered room slots."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:  # scipy is available through scikit-learn, but keep a small fallback for smoke tests.
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - exercised only in stripped environments
    linear_sum_assignment = None


@dataclass(frozen=True)
class MatchingWeights:
    centroid: float = 1.0
    area: float = 0.25
    aspect: float = 0.25
    room_type: float = 0.5
    angle: float = 0.2


DEFAULT_MATCHING_WEIGHTS = MatchingWeights()


@dataclass(frozen=True)
class MatchResult:
    target_index: torch.Tensor
    matched: torch.Tensor
    n_matched: torch.Tensor


def wrapped_angle_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(a - b + math.pi / 2.0, math.pi) - math.pi / 2.0
    return diff.abs()


def _linear_sum(cost: torch.Tensor) -> tuple[list[int], list[int]]:
    matrix = cost.detach().cpu().numpy()
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(matrix)
        return rows.tolist(), cols.tolist()

    rows: list[int] = []
    cols: list[int] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _ in range(min(matrix.shape)):
        best: tuple[float, int, int] | None = None
        for row in range(matrix.shape[0]):
            if row in used_rows:
                continue
            for col in range(matrix.shape[1]):
                if col in used_cols:
                    continue
                value = float(matrix[row, col])
                if best is None or value < best[0]:
                    best = (value, row, col)
        if best is None:
            break
        _, row, col = best
        used_rows.add(row)
        used_cols.add(col)
        rows.append(row)
        cols.append(col)
    return rows, cols


def pairwise_matching_cost(
    pred_geom: torch.Tensor,
    pred_type_logits: torch.Tensor,
    target_geom: torch.Tensor,
    target_type: torch.Tensor,
    *,
    weights: MatchingWeights = DEFAULT_MATCHING_WEIGHTS,
) -> torch.Tensor:
    """Build `[K_pred, K_target]` cost over centroid, area, aspect, type, angle."""

    eps = 1.0e-6
    centroid = torch.cdist(pred_geom[:, :2], target_geom[:, :2], p=1) / 256.0

    pred_w = pred_geom[:, 2].abs().clamp_min(eps)
    pred_h = pred_geom[:, 3].abs().clamp_min(eps)
    target_w = target_geom[:, 2].abs().clamp_min(eps)
    target_h = target_geom[:, 3].abs().clamp_min(eps)

    pred_area = (pred_w * pred_h).log()
    target_area = (target_w * target_h).log()
    area = (pred_area[:, None] - target_area[None, :]).abs()

    pred_aspect = (pred_w / pred_h).log()
    target_aspect = (target_w / target_h).log()
    aspect = (pred_aspect[:, None] - target_aspect[None, :]).abs()

    angle = wrapped_angle_distance(pred_geom[:, 4:5], target_geom[:, 4][None, :])
    angle = angle / (math.pi / 2.0)

    log_probs = pred_type_logits.log_softmax(dim=-1)
    room_type = -log_probs[:, target_type]

    return (
        weights.centroid * centroid
        + weights.area * area
        + weights.aspect * aspect
        + weights.room_type * room_type
        + weights.angle * angle
    )


def hungarian_match(
    pred_geom: torch.Tensor,
    pred_type_logits: torch.Tensor,
    target_geom: torch.Tensor,
    target_type: torch.Tensor,
    present: torch.Tensor,
    *,
    weights: MatchingWeights = DEFAULT_MATCHING_WEIGHTS,
) -> MatchResult:
    """Assign predicted slots to present target rooms for each batch item."""

    if pred_geom.ndim != 3 or pred_geom.shape[-1] != 5:
        raise ValueError("pred_geom must have shape (B, K, 5)")
    if target_geom.shape != pred_geom.shape:
        raise ValueError("target_geom must match pred_geom shape")
    if pred_type_logits.shape[:2] != pred_geom.shape[:2]:
        raise ValueError("pred_type_logits must start with shape (B, K)")
    if target_type.shape != pred_geom.shape[:2] or present.shape != pred_geom.shape[:2]:
        raise ValueError("target_type and present must have shape (B, K)")

    batch, k, _ = pred_geom.shape
    target_index = torch.full((batch, k), -1, dtype=torch.long, device=pred_geom.device)
    matched = torch.zeros((batch, k), dtype=torch.bool, device=pred_geom.device)

    for b in range(batch):
        target_slots = torch.nonzero(present[b] > 0.5, as_tuple=False).flatten()
        if target_slots.numel() == 0:
            continue
        cost = pairwise_matching_cost(
            pred_geom[b],
            pred_type_logits[b],
            target_geom[b, target_slots],
            target_type[b, target_slots],
            weights=weights,
        )
        rows, cols = _linear_sum(cost)
        if not rows:
            continue
        row_tensor = torch.as_tensor(rows, dtype=torch.long, device=pred_geom.device)
        col_tensor = torch.as_tensor(cols, dtype=torch.long, device=pred_geom.device)
        target_index[b, row_tensor] = target_slots[col_tensor]
        matched[b, row_tensor] = True

    return MatchResult(
        target_index=target_index,
        matched=matched,
        n_matched=matched.sum(dim=1),
    )
