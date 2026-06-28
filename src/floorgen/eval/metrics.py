"""Local evaluation metrics.

Two families:

  * Geometry validity + distribution metrics -- torch-free, runnable anywhere.
    These catch the failure modes the organisers named (rooms outside outline,
    overlaps, unrealistic counts) before any rendering.

  * Image-distribution metrics (FID via TorchMetrics, density/coverage via the
    vendored PRDC over Inception features) -- torch imported lazily so the
    torch-free path keeps working on machines without it. Run these on the GPU
    box alongside training.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..config import ROOM_NAMES


# ---------------------------------------------------------------------------
# Geometry validity (torch-free)
# ---------------------------------------------------------------------------
def validity_metrics(
    rooms: list[tuple[BaseGeometry, int]],
    outline: BaseGeometry,
) -> dict[str, float]:
    """Per-layout geometry health. All areas are fractions of the outline area."""
    oa = outline.area if outline.area > 0 else 1.0
    if not rooms:
        return {"outside_frac": 0.0, "overlap_frac": 0.0, "gap_frac": 1.0,
                "invalid_rate": 0.0, "n_rooms": 0}

    raw_polys = [p for p, _ in rooms]
    invalid = sum(0 if p.is_valid else 1 for p in raw_polys)
    # Repair invalid geometries for area computation (buffer(0) is standard fix)
    polys = [p.buffer(0) if not p.is_valid else p for p in raw_polys]

    union = unary_union(polys)
    covered = union.intersection(outline).area
    outside = sum(max(p.difference(outline).area, 0.0) for p in polys)
    overlap = sum(p.area for p in polys) - union.area
    gap = max(oa - covered, 0.0)

    return {
        "outside_frac": outside / oa,
        "overlap_frac": max(overlap, 0.0) / oa,
        "gap_frac": gap / oa,
        "invalid_rate": invalid / len(rooms),
        "n_rooms": len(rooms),
    }


def distribution_metrics(layouts: list[list[tuple[BaseGeometry, int]]]) -> dict:
    """Population-level distribution fit: room counts and label frequencies.
    Compare generated vs real with these to detect mode collapse / label drift."""
    counts = [len(lay) for lay in layouts]
    label_hist: Counter = Counter()
    for lay in layouts:
        for _, idx in lay:
            label_hist[ROOM_NAMES[idx]] += 1
    total = sum(label_hist.values()) or 1
    return {
        "room_count_mean": float(np.mean(counts)) if counts else 0.0,
        "room_count_std": float(np.std(counts)) if counts else 0.0,
        "label_freq": {k: v / total for k, v in label_hist.most_common()},
    }


# ---------------------------------------------------------------------------
# Image-distribution metrics (torch, imported lazily)
# ---------------------------------------------------------------------------
def inception_features(images: np.ndarray) -> np.ndarray:
    """Extract InceptionV3 pooled features for a stack of (N,H,W,3) uint8 images.
    Lazily imports torch + torchmetrics' inception. Used for PRDC and as a check
    against the FID feature space."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    inc = fid.inception
    x = torch.from_numpy(images.astype(np.uint8, copy=False)).permute(0, 3, 1, 2)
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            batch = torch.nn.functional.interpolate(
                x[i:i + 32], size=(299, 299), mode="bilinear", align_corners=False)
            feats.append(inc(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """TorchMetrics FID between two stacks of (N,H,W,3) uint8 images."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True)

    def _add(imgs: np.ndarray, real: bool) -> None:
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
        for i in range(0, len(x), 32):
            fid.update(x[i:i + 32], real=real)

    _add(real_images, True)
    _add(fake_images, False)
    return float(fid.compute())
