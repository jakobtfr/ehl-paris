"""Rendered real-vs-generated image metrics for local scoring reports."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import compute_fid, inception_features
from .prdc import compute_prdc


def image_distribution_report(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    *,
    prdc_k: int = 5,
) -> dict[str, Any]:
    """Compute FID and PRDC metrics for rendered layout image stacks."""

    real_images = np.asarray(real_images)
    fake_images = np.asarray(fake_images)
    if real_images.ndim != 4 or fake_images.ndim != 4:
        raise ValueError("real_images and fake_images must have shape (N, H, W, C)")
    if len(real_images) == 0 or len(fake_images) == 0:
        raise ValueError("real_images and fake_images must both be non-empty")
    if real_images.shape[1:] != fake_images.shape[1:]:
        raise ValueError(
            f"real/fake image shapes differ: {real_images.shape[1:]} vs {fake_images.shape[1:]}"
        )
    if prdc_k <= 0:
        raise ValueError("prdc_k must be positive")
    if len(real_images) <= 1 or len(fake_images) <= 1:
        raise ValueError("FID/PRDC need at least two real and two generated images")

    fid = compute_fid(real_images, fake_images)
    real_features = inception_features(real_images)
    fake_features = inception_features(fake_images)
    effective_k = min(prdc_k, len(real_features) - 1, len(fake_features) - 1)
    prdc = compute_prdc(real_features, fake_features, k=effective_k)
    return {
        "status": "ok",
        "n_real": int(len(real_images)),
        "n_generated": int(len(fake_images)),
        "prdc_k": int(effective_k),
        "fid": float(fid),
        "precision": float(prdc["precision"]),
        "recall": float(prdc["recall"]),
        "density": float(prdc["density"]),
        "coverage": float(prdc["coverage"]),
    }


def try_image_distribution_report(
    real_images: np.ndarray,
    fake_images: np.ndarray,
    *,
    prdc_k: int = 5,
) -> dict[str, Any]:
    """Return image metrics or a concrete blocker without fabricating scores."""

    try:
        return image_distribution_report(real_images, fake_images, prdc_k=prdc_k)
    except Exception as exc:
        return {
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
            "n_real": int(len(real_images)) if real_images.ndim else 0,
            "n_generated": int(len(fake_images)) if fake_images.ndim else 0,
            "prdc_k": int(prdc_k),
            "fid": None,
            "precision": None,
            "recall": None,
            "density": None,
            "coverage": None,
        }
