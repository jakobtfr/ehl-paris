from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

torch = pytest.importorskip("torch")

from floorgen.config import ROOM_NAMES  # noqa: E402
from floorgen.model.network import RoomFlowModel  # noqa: E402
from floorgen.model.sampler import _select_present_slots, euler_sample  # noqa: E402


def test_euler_sampler_shapes_and_decodes_mrrs() -> None:
    torch.manual_seed(7)
    outline = Polygon([(0, 0), (8, 0), (8, 6), (0, 6)])
    model = RoomFlowModel(k=4, d_model=32, boundary_points=16)

    sample = euler_sample(
        model,
        outline,
        steps=2,
        seed=7,
        threshold=-1.0,
        boundary_points=16,
    )

    assert sample.geom.shape == (4, 5)
    assert sample.type_logits.shape == (4, len(ROOM_NAMES))
    assert sample.presence_logits.shape == (4,)
    assert sample.present.shape == (4,)
    assert len(sample.mrrs) <= 4
    for mrr in sample.mrrs:
        assert np.isfinite([mrr.cx, mrr.cy, mrr.w, mrr.h, mrr.angle]).all()
        assert mrr.w > 0
        assert mrr.h > 0


def test_presence_selection_keeps_top_slot_when_threshold_rejects_all() -> None:
    probs = torch.tensor([0.10, 0.30, 0.20])

    present = _select_present_slots(probs, threshold=0.95)

    assert present.tolist() == [False, True, False]
