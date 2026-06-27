"""Canonical generation entry point.

The evaluator/organiser calls ``generate(outline)``. We expose exactly that
symbol (one positional arg), plus a richer ``sample_layouts`` for seeding,
multi-sample diversity, and raw/ranked modes.

By default this uses the trained model when a checkpoint is available; otherwise
it falls back to the heuristic baseline so the contract always holds. The scored
submission must point ``GENERATOR`` at the trained diffusion/flow model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .baseline import baseline_sample
from .config import ROOM_NAMES, SEED
from .data.outline import largest_shell
from .repr.mrr import RepairRejected, repair_partition
from .seeding import seed_everything

# A generator backend maps (outline, rng) -> list[RoomMRR]. The model trainer
# registers the trained sampler here; until then we use the baseline.
GENERATOR: Callable[[BaseGeometry, np.random.Generator], list] = baseline_sample


def _as_records(partition: list[tuple[Polygon, int]]) -> list[dict]:
    out = []
    for poly, label_idx in partition:
        out.append({
            "label": ROOM_NAMES[label_idx],
            "label_idx": label_idx,
            "polygon": poly,
            "geojson": poly.__geo_interface__,
        })
    return out


def sample_layouts(
    outline: BaseGeometry,
    seed: int = SEED,
    n_samples: int = 1,
    mode: str = "raw",
) -> list[list[dict]]:
    """Sample one or more layouts for an outline.

    Returns a list (length n_samples) of layouts; each layout is a list of room
    records with label, polygon, and geojson. Deterministic for a fixed seed;
    successive draws within a call differ (coverage-preserving).
    """
    if isinstance(outline, MultiPolygon):
        outline = largest_shell(outline)
    if not isinstance(outline, Polygon) or outline.is_empty:
        raise ValueError("outline must be a non-empty Polygon or MultiPolygon")

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(1, n_samples)):
        partition = []
        last_error: RepairRejected | None = None
        for _attempt in range(8):
            mrrs = GENERATOR(outline, rng)
            try:
                partition = repair_partition(mrrs, outline)
                break
            except RepairRejected as exc:
                last_error = exc
                continue
        if not partition and last_error is not None:
            raise last_error
        samples.append(_as_records(partition))
    return samples


def generate(outline: BaseGeometry) -> list[dict]:
    """Challenge-canonical one-argument entry point.

    Returns the room records for a single deterministic default-seed layout.
    """
    return sample_layouts(outline, seed=SEED, n_samples=1, mode="raw")[0]
