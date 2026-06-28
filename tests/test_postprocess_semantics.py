from __future__ import annotations

import numpy as np
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from floorgen.config import ROOM_NAME_TO_IDX, ROOM_NAMES
from floorgen.postprocess import RankingConfig, rank_samples
from floorgen.repr.mrr import RoomMRR


def _grid_mrrs(outline: BaseGeometry, labels: list[int]) -> list[RoomMRR]:
    minx, miny, maxx, maxy = outline.bounds
    step = (maxx - minx) / len(labels)
    mrrs = []
    for i, label in enumerate(labels):
        x0 = minx + i * step
        x1 = x0 + step
        mrrs.append(
            RoomMRR(
                cx=(x0 + x1) / 2,
                cy=(miny + maxy) / 2,
                w=step,
                h=maxy - miny,
                angle=0.0,
                label_idx=label,
            )
        )
    return mrrs


def test_ranked_mode_calibrates_collapsed_semantic_labels() -> None:
    outline = box(0, 0, 12, 8)
    balcony = ROOM_NAME_TO_IDX["Balcony"]

    def collapsed_generator(_outline, _rng):
        return _grid_mrrs(outline, [balcony] * 8)

    selection = rank_samples(
        outline,
        collapsed_generator,
        np.random.default_rng(7),
        n_samples=1,
        config=RankingConfig(candidate_budget=1),
    )

    labels = [ROOM_NAMES[label] for _poly, label in selection.layouts[0]]
    assert len(set(labels)) >= 4
    assert selection.provenance["semantic_repair_count"] == 1
    repair = selection.provenance["selected_semantic_repairs"][0]
    assert repair["before_label_counts"] == {"Balcony": 8}
    assert repair["after_label_counts"]["Livingroom"] == 1
    assert repair["after_label_counts"]["Kitchen"] == 1
    assert repair["after_label_counts"]["Bathroom"] == 1


def test_ranked_mode_keeps_mixed_semantic_labels() -> None:
    outline = box(0, 0, 12, 8)
    labels = [
        ROOM_NAME_TO_IDX["Bedroom"],
        ROOM_NAME_TO_IDX["Kitchen"],
        ROOM_NAME_TO_IDX["Bathroom"],
        ROOM_NAME_TO_IDX["Livingroom"],
    ]

    def mixed_generator(_outline, _rng):
        return _grid_mrrs(outline, labels)

    selection = rank_samples(
        outline,
        mixed_generator,
        np.random.default_rng(7),
        n_samples=1,
        config=RankingConfig(candidate_budget=1),
    )

    generated_labels = [label for _poly, label in selection.layouts[0]]
    assert generated_labels == labels
    assert selection.provenance["semantic_repair_count"] == 0
    assert selection.provenance["selected_semantic_repairs"] == []
