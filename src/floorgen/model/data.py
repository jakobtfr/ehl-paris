"""Torch dataset for processed MSD units."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..config import MAX_ROOMS_K
from .geometry import outline_conditioning_from_record, record_to_targets


@dataclass(frozen=True)
class ModelBatch:
    outline_xy: torch.Tensor
    outline_mask: torch.Tensor
    scale: torch.Tensor
    target_geom: torch.Tensor
    target_type: torch.Tensor
    present: torch.Tensor
    unit_id: list[int]
    split: list[str]
    n_rooms: torch.Tensor
    n_truncated: torch.Tensor

    def to(self, device: torch.device | str) -> ModelBatch:
        return ModelBatch(
            outline_xy=self.outline_xy.to(device),
            outline_mask=self.outline_mask.to(device),
            scale=self.scale.to(device),
            target_geom=self.target_geom.to(device),
            target_type=self.target_type.to(device),
            present=self.present.to(device),
            unit_id=self.unit_id,
            split=self.split,
            n_rooms=self.n_rooms.to(device),
            n_truncated=self.n_truncated.to(device),
        )


class FloorRecordDataset(Dataset):
    """Dataset over `data/processed/units.jsonl` records.

    Each item contains normalized outline boundary points, six scale features,
    padded room geometry targets `[K, 5]`, room type targets `[K]`, and presence
    targets `[K]`.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        split: str | None = None,
        k: int = MAX_ROOMS_K,
        boundary_points: int = 128,
        limit: int | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"processed units JSONL not found: {self.jsonl_path}")
        if k <= 0:
            raise ValueError("k must be positive")
        if boundary_points <= 0:
            raise ValueError("boundary_points must be positive")

        self.split = split
        self.k = int(k)
        self.boundary_points = int(boundary_points)
        self.records: list[dict[str, Any]] = []

        with self.jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if split is not None and record.get("split") != split:
                    continue
                self.records.append(record)
                if limit is not None and len(self.records) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        conditioning = outline_conditioning_from_record(
            record,
            boundary_points=self.boundary_points,
        )
        targets = record_to_targets(record, k=self.k)
        return {
            "outline_xy": torch.as_tensor(conditioning.outline_xy, dtype=torch.float32),
            "outline_mask": torch.as_tensor(conditioning.outline_mask, dtype=torch.bool),
            "scale": torch.as_tensor(conditioning.scale, dtype=torch.float32),
            "target_geom": torch.as_tensor(targets.target_geom, dtype=torch.float32),
            "target_type": torch.as_tensor(targets.target_type, dtype=torch.long),
            "present": torch.as_tensor(targets.present, dtype=torch.float32),
            "unit_id": int(record["unit_id"]),
            "split": str(record.get("split", "")),
            "n_rooms": int(record["n_rooms"]),
            "n_truncated": int(targets.n_truncated),
        }


def collate_model_batch(items: Sequence[dict[str, Any]]) -> ModelBatch:
    if not items:
        raise ValueError("cannot collate an empty model batch")
    return ModelBatch(
        outline_xy=torch.stack([item["outline_xy"] for item in items]),
        outline_mask=torch.stack([item["outline_mask"] for item in items]),
        scale=torch.stack([item["scale"] for item in items]),
        target_geom=torch.stack([item["target_geom"] for item in items]),
        target_type=torch.stack([item["target_type"] for item in items]),
        present=torch.stack([item["present"] for item in items]),
        unit_id=[int(item["unit_id"]) for item in items],
        split=[str(item["split"]) for item in items],
        n_rooms=torch.tensor([int(item["n_rooms"]) for item in items], dtype=torch.long),
        n_truncated=torch.tensor(
            [int(item["n_truncated"]) for item in items],
            dtype=torch.long,
        ),
    )
