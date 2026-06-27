from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from shapely.geometry import Polygon

pytest.importorskip("torch")

from floorgen.data.normalize import fit_transform, scale_features  # noqa: E402
from floorgen.model.data import FloorRecordDataset, collate_model_batch  # noqa: E402
from floorgen.model.geometry import (
    outline_conditioning_from_record,
    record_to_targets,
    sample_boundary_points,
)  # noqa: E402


def make_record(unit_id: int = 1, split: str = "train") -> dict:
    outline = Polygon([(0, 0), (8, 0), (8, 6), (0, 6)])
    rooms = [
        ("Livingroom", 1, Polygon([(0, 0), (5, 0), (5, 6), (0, 6)])),
        ("Kitchen", 2, Polygon([(5, 0), (8, 0), (8, 3), (5, 3)])),
        ("Bathroom", 7, Polygon([(5, 3), (8, 3), (8, 6), (5, 6)])),
    ]
    transform = fit_transform(outline)
    return {
        "unit_id": unit_id,
        "plan_id": 10,
        "floor_id": 10,
        "n_rooms": len(rooms),
        "outline_wkt": outline.wkt,
        "transform": asdict(transform),
        "scale_features": asdict(scale_features(outline)),
        "rooms": [
            {
                "label": label,
                "label_idx": label_idx,
                "wkt": poly.wkt,
                "area_m2": float(poly.area),
            }
            for label, label_idx, poly in rooms
        ],
        "split": split,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_boundary_sampling_returns_fixed_points() -> None:
    outline = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    xy, mask = sample_boundary_points(outline, 12)
    assert xy.shape == (12, 2)
    assert mask.shape == (12,)
    assert mask.all()


def test_record_to_targets_pads_fixed_slots() -> None:
    targets = record_to_targets(make_record(), k=5)
    assert targets.target_geom.shape == (5, 5)
    assert targets.target_type.shape == (5,)
    assert targets.present.shape == (5,)
    assert targets.present.sum() == 3
    assert targets.n_truncated == 0
    assert (targets.target_geom[:3, 2:4] > 0).all()


def test_outline_conditioning_shapes() -> None:
    conditioning = outline_conditioning_from_record(make_record(), boundary_points=16)
    assert conditioning.outline_xy.shape == (16, 2)
    assert conditioning.outline_mask.shape == (16,)
    assert conditioning.scale.shape == (6,)
    assert conditioning.outline_mask.all()


def test_floor_record_dataset_and_collate(tmp_path: Path) -> None:
    jsonl = tmp_path / "units.jsonl"
    write_jsonl(jsonl, [make_record(1, "train"), make_record(2, "val")])

    dataset = FloorRecordDataset(jsonl, split="train", k=4, boundary_points=10)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["outline_xy"].shape == (10, 2)
    assert item["target_geom"].shape == (4, 5)
    assert item["present"].sum().item() == 3

    batch = collate_model_batch([item, item])
    assert batch.outline_xy.shape == (2, 10, 2)
    assert batch.scale.shape == (2, 6)
    assert batch.target_geom.shape == (2, 4, 5)
    assert batch.target_type.shape == (2, 4)
    assert batch.present.shape == (2, 4)
    assert batch.unit_id == [1, 1]


def test_real_processed_data_smoke_if_available() -> None:
    jsonl = Path("data/processed/units.jsonl")
    if not jsonl.exists():
        pytest.skip("real processed data is not available")

    dataset = FloorRecordDataset(jsonl, split="train", k=24, boundary_points=12, limit=2)
    assert len(dataset) == 2
    batch = collate_model_batch([dataset[0], dataset[1]])
    assert batch.outline_xy.shape == (2, 12, 2)
    assert batch.target_geom.shape == (2, 24, 5)
    assert batch.present.sum().item() > 0
