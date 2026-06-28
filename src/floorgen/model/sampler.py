"""Euler sampler and checkpoint loader for the flow model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from shapely.geometry.base import BaseGeometry

from ..config import MAX_ROOMS_K, ROOM_NAMES
from ..repr.mrr import RoomMRR
from .geometry import decode_mrr_slots, outline_conditioning_from_geometry
from .network import RoomFlowModel, RoomFlowTransformer


@dataclass(frozen=True)
class SampleOutput:
    geom: torch.Tensor
    type_logits: torch.Tensor
    presence_logits: torch.Tensor
    present: torch.Tensor
    mrrs: list[RoomMRR]


def _model_device(model: RoomFlowModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _select_present_slots(presence_prob: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold presence logits, with a model-ranked non-empty fallback."""

    if presence_prob.ndim != 1:
        raise ValueError(f"presence_prob must have shape (K,), got {presence_prob.shape}")
    present = presence_prob > threshold
    if bool(present.any()):
        return present

    expected = int(
        torch.clamp(
            torch.round(presence_prob.sum()),
            min=1,
            max=presence_prob.numel(),
        ).item()
    )
    keep = torch.topk(presence_prob, k=expected).indices
    present = torch.zeros_like(presence_prob, dtype=torch.bool)
    present[keep] = True
    return present


@torch.no_grad()
def euler_sample(
    model: RoomFlowModel,
    outline: BaseGeometry,
    *,
    steps: int = 32,
    seed: int | None = None,
    threshold: float = 0.5,
    boundary_points: int | None = None,
    device: torch.device | str | None = None,
) -> SampleOutput:
    """Integrate the learned velocity field from noise to normalized MRR slots."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(device) if device is not None else _model_device(model)
    model.eval()
    model.to(device)
    boundary_points = boundary_points or model.boundary_points

    conditioning = outline_conditioning_from_geometry(outline, boundary_points=boundary_points)
    outline_xy = torch.as_tensor(conditioning.outline_xy, dtype=torch.float32, device=device)[None]
    outline_mask = torch.as_tensor(conditioning.outline_mask, dtype=torch.bool, device=device)[None]
    scale = torch.as_tensor(conditioning.scale, dtype=torch.float32, device=device)[None]

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    x = torch.randn((1, model.k, 5), generator=generator, device=device)
    dt = 1.0 / float(steps)
    last_output = None
    for i in range(steps):
        t = torch.full((1,), i / float(steps), dtype=torch.float32, device=device)
        last_output = model(x, t, outline_xy, scale, outline_mask)
        x = x + dt * last_output.velocity

    final_t = torch.ones((1,), dtype=torch.float32, device=device)
    last_output = model(x, final_t, outline_xy, scale, outline_mask)
    type_idx = last_output.type_logits.argmax(dim=-1)[0]
    presence_prob = last_output.presence_logits.sigmoid()[0]
    present = _select_present_slots(presence_prob, threshold)
    mrrs = decode_mrr_slots(
        x[0].detach().cpu().numpy(),
        type_idx.detach().cpu().numpy(),
        present.detach().cpu().numpy(),
        conditioning.transform,
    )
    return SampleOutput(
        geom=x[0].detach().cpu(),
        type_logits=last_output.type_logits[0].detach().cpu(),
        presence_logits=last_output.presence_logits[0].detach().cpu(),
        present=present.detach().cpu(),
        mrrs=mrrs,
    )


def load_generator(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    steps: int = 32,
    threshold: float = 0.5,
) -> Callable[[BaseGeometry, np.random.Generator], list[RoomMRR]]:
    """Load a checkpoint as a `floorgen.generate.GENERATOR`-compatible callable."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    architecture = config.get("architecture", "mlp")
    checkpoint_labels = tuple(checkpoint.get("label_names") or ROOM_NAMES)
    if checkpoint_labels != ROOM_NAMES:
        raise ValueError(
            "checkpoint label_names do not match floorgen.config.ROOM_NAMES: "
            f"{checkpoint_labels!r} != {ROOM_NAMES!r}"
        )

    if architecture == "transformer":
        model = RoomFlowTransformer(
            num_types=int(config.get("num_types", len(ROOM_NAMES))),
            k=int(config.get("k", MAX_ROOMS_K)),
            d_model=int(config.get("d_model", 512)),
            boundary_points=int(config.get("boundary_points", 128)),
            num_layers=int(config.get("num_layers", 4)),
            nhead=int(config.get("nhead", 8)),
            dim_feedforward=int(config.get("dim_feedforward", 2048)),
            dropout=float(config.get("dropout", 0.0)),
        )
    else:
        model = RoomFlowModel(
            num_types=int(config.get("num_types", len(ROOM_NAMES))),
            k=int(config.get("k", MAX_ROOMS_K)),
            d_model=int(config.get("d_model", 128)),
            boundary_points=int(config.get("boundary_points", 128)),
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    def _generate(outline: BaseGeometry, rng: np.random.Generator) -> list[RoomMRR]:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return euler_sample(
            model,
            outline,
            steps=steps,
            seed=seed,
            threshold=threshold,
            device=device,
        ).mrrs

    return _generate
