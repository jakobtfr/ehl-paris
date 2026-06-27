from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from floorgen.config import ROOM_NAMES  # noqa: E402
from floorgen.model.data import ModelBatch  # noqa: E402
from floorgen.model.losses import conditional_flow_matching_loss  # noqa: E402
from floorgen.model.matching import hungarian_match  # noqa: E402
from floorgen.model.network import RoomFlowModel  # noqa: E402


def make_batch(batch_size: int = 2, k: int = 4, points: int = 8) -> ModelBatch:
    target_geom = torch.zeros(batch_size, k, 5)
    target_geom[:, 0] = torch.tensor([0.0, 0.0, 4.0, 2.0, 0.0])
    target_geom[:, 1] = torch.tensor([8.0, 0.0, 3.0, 2.0, 0.2])
    target_type = torch.zeros(batch_size, k, dtype=torch.long)
    target_type[:, 0] = 1
    target_type[:, 1] = 2
    present = torch.zeros(batch_size, k)
    present[:, :2] = 1.0
    return ModelBatch(
        outline_xy=torch.randn(batch_size, points, 2),
        outline_mask=torch.ones(batch_size, points, dtype=torch.bool),
        scale=torch.randn(batch_size, 6),
        target_geom=target_geom,
        target_type=target_type,
        present=present,
        unit_id=list(range(batch_size)),
        split=["train"] * batch_size,
        n_rooms=torch.full((batch_size,), 2, dtype=torch.long),
        n_truncated=torch.zeros(batch_size, dtype=torch.long),
    )


def test_room_flow_model_forward_shapes() -> None:
    batch = make_batch()
    model = RoomFlowModel(k=4, d_model=32, boundary_points=8)
    x_t = torch.randn(2, 4, 5)
    t = torch.rand(2)
    output = model(x_t, t, batch.outline_xy, batch.scale, batch.outline_mask)
    assert output.velocity.shape == (2, 4, 5)
    assert output.type_logits.shape == (2, 4, len(ROOM_NAMES))
    assert output.presence_logits.shape == (2, 4)


def test_hungarian_matching_swaps_unordered_slots() -> None:
    target_geom = torch.zeros(1, 4, 5)
    target_geom[0, 0] = torch.tensor([0.0, 0.0, 4.0, 2.0, 0.0])
    target_geom[0, 1] = torch.tensor([10.0, 0.0, 3.0, 2.0, 0.0])
    pred_geom = target_geom.clone()
    pred_geom[0, 0] = target_geom[0, 1]
    pred_geom[0, 1] = target_geom[0, 0]
    target_type = torch.tensor([[1, 2, 0, 0]])
    present = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    logits = torch.full((1, 4, len(ROOM_NAMES)), -4.0)
    logits[0, 0, 2] = 4.0
    logits[0, 1, 1] = 4.0

    match = hungarian_match(pred_geom, logits, target_geom, target_type, present)
    assert match.target_index[0, 0].item() == 1
    assert match.target_index[0, 1].item() == 0
    assert match.matched[0, :2].all()
    assert match.n_matched.tolist() == [2]


def test_conditional_flow_loss_is_finite_and_trains_one_step() -> None:
    torch.manual_seed(123)
    batch = make_batch()
    model = RoomFlowModel(k=4, d_model=32, boundary_points=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    result = conditional_flow_matching_loss(model, batch)
    assert torch.isfinite(result.total)
    assert result.n_matched.tolist() == [2, 2]

    optimizer.zero_grad()
    result.total.backward()
    optimizer.step()

    second = conditional_flow_matching_loss(model, batch)
    assert torch.isfinite(second.total)
