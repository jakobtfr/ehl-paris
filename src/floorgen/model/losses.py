"""Conditional flow-matching objective for fixed-slot room tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .data import ModelBatch
from .matching import DEFAULT_MATCHING_WEIGHTS, MatchingWeights, hungarian_match
from .network import RoomFlowModel


@dataclass(frozen=True)
class FlowLossWeights:
    velocity: float = 1.0
    room_type: float = 0.25
    presence: float = 0.25
    null_velocity: float = 0.02


DEFAULT_FLOW_LOSS_WEIGHTS = FlowLossWeights()


@dataclass(frozen=True)
class FlowLossResult:
    total: torch.Tensor
    velocity: torch.Tensor
    room_type: torch.Tensor
    presence: torch.Tensor
    null_velocity: torch.Tensor
    n_matched: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "velocity": float(self.velocity.detach().cpu()),
            "room_type": float(self.room_type.detach().cpu()),
            "presence": float(self.presence.detach().cpu()),
            "null_velocity": float(self.null_velocity.detach().cpu()),
            "n_matched": float(self.n_matched.float().mean().detach().cpu()),
        }


def _gather_matched_targets(
    batch: ModelBatch,
    target_index: torch.Tensor,
    matched: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_index = target_index.clamp_min(0)
    matched_geom = batch.target_geom.gather(
        1,
        safe_index[..., None].expand(-1, -1, batch.target_geom.shape[-1]),
    )
    matched_geom = torch.where(matched[..., None], matched_geom, torch.zeros_like(matched_geom))
    matched_type = batch.target_type.gather(1, safe_index)
    matched_type = torch.where(matched, matched_type, torch.zeros_like(matched_type))
    return matched_geom, matched_type


def conditional_flow_matching_loss(
    model: RoomFlowModel,
    batch: ModelBatch,
    *,
    loss_weights: FlowLossWeights = DEFAULT_FLOW_LOSS_WEIGHTS,
    matching_weights: MatchingWeights = DEFAULT_MATCHING_WEIGHTS,
) -> FlowLossResult:
    """Sample a linear flow path and train velocity, type, and presence heads."""

    target_geom = batch.target_geom
    batch_size = target_geom.shape[0]
    z0 = torch.randn_like(target_geom)
    t = torch.rand((batch_size,), dtype=target_geom.dtype, device=target_geom.device)
    t_view = t[:, None, None]
    x_t = (1.0 - t_view) * z0 + t_view * target_geom

    output = model(x_t, t, batch.outline_xy, batch.scale, batch.outline_mask)
    predicted_x1 = x_t + (1.0 - t_view) * output.velocity
    matches = hungarian_match(
        predicted_x1.detach(),
        output.type_logits.detach(),
        batch.target_geom,
        batch.target_type,
        batch.present,
        weights=matching_weights,
    )
    matched_geom, matched_type = _gather_matched_targets(
        batch,
        matches.target_index,
        matches.matched,
    )

    matched_float = matches.matched.float()
    velocity_target = matched_geom - z0
    velocity_per_slot = (output.velocity - velocity_target).pow(2).mean(dim=-1)
    velocity_loss = (velocity_per_slot * matched_float).sum() / matched_float.sum().clamp_min(1.0)

    if matches.matched.any():
        room_type_loss = F.cross_entropy(output.type_logits[matches.matched], matched_type[matches.matched])
    else:
        room_type_loss = output.type_logits.sum() * 0.0

    presence_loss = F.binary_cross_entropy_with_logits(
        output.presence_logits,
        matched_float,
    )
    null_mask = ~matches.matched
    if null_mask.any():
        null_velocity_loss = output.velocity[null_mask].pow(2).mean()
    else:
        null_velocity_loss = output.velocity.sum() * 0.0

    total = (
        loss_weights.velocity * velocity_loss
        + loss_weights.room_type * room_type_loss
        + loss_weights.presence * presence_loss
        + loss_weights.null_velocity * null_velocity_loss
    )
    return FlowLossResult(
        total=total,
        velocity=velocity_loss,
        room_type=room_type_loss,
        presence=presence_loss,
        null_velocity=null_velocity_loss,
        n_matched=matches.n_matched,
    )
