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
from .postprocess import RankingConfig, rank_samples
from .repr.mrr import RepairRejected, repair_partition
from .seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "flow-transformer-amd-862d422.pt"
MLP_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "flow-full-8303584.pt"
LOCAL_TRANSFORMER_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "flow-transformer-862d422.pt"
DEFAULT_MODEL_ALIAS = "amd-transformer"
MODEL_CHECKPOINTS = {
    "amd-transformer": DEFAULT_CHECKPOINT_PATH,
    "amd": DEFAULT_CHECKPOINT_PATH,
    "transformer-amd": DEFAULT_CHECKPOINT_PATH,
    "local-transformer": LOCAL_TRANSFORMER_CHECKPOINT_PATH,
    "mlp": MLP_CHECKPOINT_PATH,
    "legacy-mlp": MLP_CHECKPOINT_PATH,
}
DEFAULT_DEVICE = "auto"
DEFAULT_SAMPLE_STEPS = "16"
DEFAULT_PRESENCE_THRESHOLD = "0.5"
DEFAULT_GENERATION_MODE = "ranked"
DEFAULT_CANDIDATE_BUDGET = "16"


def _nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _default_checkpoint_enabled() -> bool:
    value = (_nonempty_env("FLOORGEN_DISABLE_DEFAULT_CHECKPOINT") or "").lower()
    return value not in {"1", "true", "yes", "on"}


def default_device() -> str:
    """Prefer the best local inference device for the AMD checkpoint."""

    explicit = _nonempty_env("FLOORGEN_DEVICE")
    if explicit and explicit.lower() != "auto":
        return explicit

    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _normalize_model_alias(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def resolve_checkpoint_reference(value: str) -> str:
    """Resolve a model alias or user path to a checkpoint path string."""

    alias = _normalize_model_alias(value)
    if alias in MODEL_CHECKPOINTS:
        return str(MODEL_CHECKPOINTS[alias])

    path = Path(value).expanduser()
    if not path.is_absolute() and not path.exists():
        repo_path = REPO_ROOT / path
        if repo_path.exists():
            return str(repo_path)
    return str(path)


def _selected_model_alias() -> str:
    return _normalize_model_alias(_nonempty_env("FLOORGEN_MODEL") or DEFAULT_MODEL_ALIAS)


def _model_alias_for_checkpoint(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    try:
        path = Path(checkpoint).expanduser().resolve()
    except Exception:
        path = Path(checkpoint)
    for alias, candidate in MODEL_CHECKPOINTS.items():
        try:
            if candidate.resolve() == path:
                return alias
        except Exception:
            if candidate == path:
                return alias
    return None


def _checkpoint_path() -> str | None:
    explicit = _nonempty_env("FLOORGEN_CHECKPOINT")
    if explicit:
        return resolve_checkpoint_reference(explicit)
    if _default_checkpoint_enabled():
        checkpoint = MODEL_CHECKPOINTS.get(_selected_model_alias())
        if checkpoint and checkpoint.exists():
            return str(checkpoint)
    return None


def _generator_from_env() -> Callable[[BaseGeometry, np.random.Generator], list] | None:
    checkpoint = _checkpoint_path()
    if not checkpoint:
        return None

    steps = int(os.environ.get("FLOORGEN_SAMPLE_STEPS", DEFAULT_SAMPLE_STEPS))
    threshold = float(os.environ.get("FLOORGEN_PRESENCE_THRESHOLD", DEFAULT_PRESENCE_THRESHOLD))
    if steps <= 0:
        raise ValueError("FLOORGEN_SAMPLE_STEPS must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("FLOORGEN_PRESENCE_THRESHOLD must be between 0 and 1")
    device = default_device()

    from .model.sampler import load_generator

    return load_generator(Path(checkpoint), device=device, steps=steps, threshold=threshold)


# A generator backend maps (outline, rng) -> list[RoomMRR]. The Python symbol is
# initialized to the baseline so tests and custom callers can replace it, but
# the active runtime prefers the AMD Transformer checkpoint when it exists
# locally. The baseline is therefore a missing-artifact fallback, not the normal
# judged path.
GENERATOR: Callable[[BaseGeometry, np.random.Generator], list] = baseline_sample
_ENV_GENERATOR_KEY: tuple[str, str, str, str] | None = None
_ENV_GENERATOR: Callable[[BaseGeometry, np.random.Generator], list] | None = None
LAST_RANKING_PROVENANCE: dict | None = None


def _active_generator() -> Callable[[BaseGeometry, np.random.Generator], list]:
    global _ENV_GENERATOR
    global _ENV_GENERATOR_KEY

    if GENERATOR is not baseline_sample:
        return GENERATOR
    checkpoint = _checkpoint_path()
    if not checkpoint:
        return GENERATOR
    key = (
        checkpoint,
        default_device(),
        os.environ.get("FLOORGEN_SAMPLE_STEPS", DEFAULT_SAMPLE_STEPS),
        os.environ.get("FLOORGEN_PRESENCE_THRESHOLD", DEFAULT_PRESENCE_THRESHOLD),
    )
    if _ENV_GENERATOR is None or key != _ENV_GENERATOR_KEY:
        _ENV_GENERATOR = _generator_from_env()
        _ENV_GENERATOR_KEY = key
    return _ENV_GENERATOR or GENERATOR


def backend_provenance() -> dict[str, str | None]:
    """Describe the currently configured generation backend for reports/demos."""

    mode = os.environ.get("FLOORGEN_GENERATION_MODE", DEFAULT_GENERATION_MODE)
    candidate_budget = os.environ.get("FLOORGEN_CANDIDATE_BUDGET", DEFAULT_CANDIDATE_BUDGET)
    if GENERATOR is not baseline_sample:
        return {
            "backend": "custom-generator",
            "model": None,
            "checkpoint": None,
            "device": None,
            "steps": None,
            "presence_threshold": None,
            "generation_mode": mode,
            "candidate_budget": candidate_budget,
        }
    checkpoint = _checkpoint_path()
    if checkpoint:
        return {
            "backend": "flow-checkpoint",
            "model": _model_alias_for_checkpoint(checkpoint),
            "checkpoint": checkpoint,
            "device": default_device(),
            "steps": os.environ.get("FLOORGEN_SAMPLE_STEPS", DEFAULT_SAMPLE_STEPS),
            "presence_threshold": os.environ.get(
                "FLOORGEN_PRESENCE_THRESHOLD",
                DEFAULT_PRESENCE_THRESHOLD,
            ),
            "generation_mode": mode,
            "candidate_budget": candidate_budget,
        }
    return {
        "backend": "baseline",
        "model": None,
        "checkpoint": None,
        "device": None,
        "steps": None,
        "presence_threshold": None,
        "generation_mode": mode,
        "candidate_budget": candidate_budget,
    }


def _generation_mode_from_env() -> str:
    mode = os.environ.get("FLOORGEN_GENERATION_MODE", DEFAULT_GENERATION_MODE).lower().strip()
    if mode not in {"raw", "ranked"}:
        raise ValueError("FLOORGEN_GENERATION_MODE must be 'raw' or 'ranked'")
    return mode


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
    mode: str | None = None,
    candidate_budget: int | None = None,
) -> list[list[dict]]:
    """Sample one or more layouts for an outline.

    Returns a list (length n_samples) of layouts; each layout is a list of room
    records with label, polygon, and geojson. Deterministic for a fixed seed;
    successive draws within a call differ (coverage-preserving).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if isinstance(outline, MultiPolygon):
        outline = largest_shell(outline)
    if not isinstance(outline, Polygon) or outline.is_empty:
        raise ValueError("outline must be a non-empty Polygon or MultiPolygon")

    normalized_mode = (mode or _generation_mode_from_env()).lower().strip()
    if normalized_mode not in {"raw", "ranked"}:
        raise ValueError("mode must be 'raw' or 'ranked'")

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    n_samples = max(1, n_samples)

    global LAST_RANKING_PROVENANCE
    if normalized_mode == "ranked":
        budget = candidate_budget
        if budget is None:
            budget = int(os.environ.get("FLOORGEN_CANDIDATE_BUDGET", DEFAULT_CANDIDATE_BUDGET))
            budget = max(budget, n_samples)
        if budget <= 0:
            raise ValueError("candidate_budget must be positive")
        selection = rank_samples(
            outline,
            _active_generator(),
            rng,
            n_samples=n_samples,
            config=RankingConfig(candidate_budget=budget),
        )
        LAST_RANKING_PROVENANCE = selection.provenance
        return [_as_records(partition) for partition in selection.layouts]

    LAST_RANKING_PROVENANCE = None
    samples = []
    for _ in range(n_samples):
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
    return sample_layouts(outline, seed=SEED, n_samples=1, mode=_generation_mode_from_env())[0]
