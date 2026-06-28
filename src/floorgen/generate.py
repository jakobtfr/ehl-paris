"""Canonical generation entry point.

The evaluator/organiser calls ``generate(outline)``. We expose exactly that
symbol (one positional arg), plus a richer ``sample_layouts`` for seeding,
multi-sample diversity, and raw/ranked modes.

By default this uses the trained model when a checkpoint is available; otherwise
it falls back to the heuristic baseline so the contract always holds. The scored
submission must point ``GENERATOR`` at the trained diffusion/flow model.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .baseline import baseline_sample
from .config import ROOM_NAMES, SEED
from .data.outline import largest_shell
from .repr.mrr import RepairRejected, repair_partition
from .seeding import seed_everything


def _generator_from_env() -> Callable[[BaseGeometry, np.random.Generator], list] | None:
    checkpoint = os.environ.get("FLOORGEN_CHECKPOINT")
    if not checkpoint:
        return None

    steps = int(os.environ.get("FLOORGEN_SAMPLE_STEPS", "32"))
    threshold = float(os.environ.get("FLOORGEN_PRESENCE_THRESHOLD", "0.5"))
    if steps <= 0:
        raise ValueError("FLOORGEN_SAMPLE_STEPS must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("FLOORGEN_PRESENCE_THRESHOLD must be between 0 and 1")
    device = os.environ.get("FLOORGEN_DEVICE", "cpu")

    from .model.sampler import load_generator

    return load_generator(Path(checkpoint), device=device, steps=steps, threshold=threshold)


# A generator backend maps (outline, rng) -> list[RoomMRR]. By default this is
# the baseline. If FLOORGEN_CHECKPOINT is set and GENERATOR has not been
# explicitly replaced, the checkpoint-backed generator is loaded lazily on first
# generation so CLIs with explicit --checkpoint flags can parse their arguments
# without being preempted by stale environment state.
GENERATOR: Callable[[BaseGeometry, np.random.Generator], list] = baseline_sample
_ENV_GENERATOR_KEY: tuple[str, str, str, str] | None = None
_ENV_GENERATOR: Callable[[BaseGeometry, np.random.Generator], list] | None = None


def _active_generator() -> Callable[[BaseGeometry, np.random.Generator], list]:
    global _ENV_GENERATOR
    global _ENV_GENERATOR_KEY

    if GENERATOR is not baseline_sample:
        return GENERATOR
    checkpoint = os.environ.get("FLOORGEN_CHECKPOINT")
    if not checkpoint:
        return GENERATOR
    key = (
        checkpoint,
        os.environ.get("FLOORGEN_DEVICE", "cpu"),
        os.environ.get("FLOORGEN_SAMPLE_STEPS", "32"),
        os.environ.get("FLOORGEN_PRESENCE_THRESHOLD", "0.5"),
    )
    if _ENV_GENERATOR is None or key != _ENV_GENERATOR_KEY:
        _ENV_GENERATOR = _generator_from_env()
        _ENV_GENERATOR_KEY = key
    return _ENV_GENERATOR or GENERATOR


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
            mrrs = _active_generator()(outline, rng)
            try:
                partition = repair_partition(mrrs, outline)
                if not partition:
                    last_error = RepairRejected("generator produced no repairable rooms")
                    continue
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
