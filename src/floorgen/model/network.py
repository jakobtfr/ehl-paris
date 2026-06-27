"""Small conditional flow network for fixed-slot MRR tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..config import MAX_ROOMS_K, ROOM_NAMES


@dataclass(frozen=True)
class ModelOutput:
    velocity: torch.Tensor
    type_logits: torch.Tensor
    presence_logits: torch.Tensor


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
    half = dim // 2
    if half == 0:
        return t[:, None]
    frequencies = torch.exp(
        torch.arange(half, device=t.device, dtype=t.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = t[:, None] * frequencies[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RoomFlowModel(nn.Module):
    """Predict flow velocity, room type, and presence per fixed room slot."""

    def __init__(
        self,
        *,
        num_types: int = len(ROOM_NAMES),
        k: int = MAX_ROOMS_K,
        d_model: int = 128,
        boundary_points: int = 128,
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        self.num_types = int(num_types)
        self.k = int(k)
        self.d_model = int(d_model)
        self.boundary_points = int(boundary_points)

        self.point_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(6, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.geom_mlp = nn.Sequential(
            nn.Linear(5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.slot_embedding = nn.Embedding(k, d_model)
        self.slot_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.velocity_head = nn.Linear(d_model, 5)
        self.type_head = nn.Linear(d_model, num_types)
        self.presence_head = nn.Linear(d_model, 1)

    def encode_outline(
        self,
        outline_xy: torch.Tensor,
        scale: torch.Tensor,
        outline_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if outline_xy.ndim != 3 or outline_xy.shape[-1] != 2:
            raise ValueError(f"outline_xy must have shape (B, P, 2), got {outline_xy.shape}")
        if scale.ndim != 2 or scale.shape[-1] != 6:
            raise ValueError(f"scale must have shape (B, 6), got {scale.shape}")
        if outline_xy.shape[0] != scale.shape[0]:
            raise ValueError("outline and scale batch sizes differ")
        if outline_mask is None:
            outline_mask = torch.ones(
                outline_xy.shape[:2],
                dtype=torch.bool,
                device=outline_xy.device,
            )
        if outline_mask.shape != outline_xy.shape[:2]:
            raise ValueError("outline_mask must have shape (B, P)")

        point_features = self.point_mlp(outline_xy)
        masked = point_features.masked_fill(~outline_mask[..., None], -1.0e9)
        pooled = masked.max(dim=1).values
        has_point = outline_mask.any(dim=1)
        pooled = torch.where(has_point[:, None], pooled, torch.zeros_like(pooled))
        return pooled + self.scale_mlp(scale)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        outline_xy: torch.Tensor,
        scale: torch.Tensor,
        outline_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        if x_t.ndim != 3 or x_t.shape[1:] != (self.k, 5):
            raise ValueError(f"x_t must have shape (B, {self.k}, 5), got {x_t.shape}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t must have shape ({x_t.shape[0]},), got {t.shape}")

        condition = self.encode_outline(outline_xy, scale, outline_mask)
        time = self.time_mlp(sinusoidal_time_embedding(t, self.d_model))
        geom = self.geom_mlp(x_t)
        slot_ids = torch.arange(self.k, device=x_t.device)
        slots = self.slot_embedding(slot_ids)[None, :, :].expand(x_t.shape[0], -1, -1)
        context = (condition + time)[:, None, :].expand(-1, self.k, -1)
        hidden = self.slot_mlp(torch.cat([geom, context, slots], dim=-1))
        return ModelOutput(
            velocity=self.velocity_head(hidden),
            type_logits=self.type_head(hidden),
            presence_logits=self.presence_head(hidden).squeeze(-1),
        )
