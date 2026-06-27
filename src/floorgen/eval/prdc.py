"""Density and Coverage (and precision/recall) from clovaai/generative-evaluation-prdc.

Vendored, numpy-only implementation of the metrics the organisers use. Kept
faithful to the reference so local scores track the official harness. Operates
on feature vectors (e.g. InceptionV3 pooled features); the feature extractor is
provided separately so this file stays torch-free.

Reference: Naeem et al., "Reliable Fidelity and Diversity Metrics for
Generative Models" (ICML 2020).
"""

from __future__ import annotations

import numpy as np


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distances; uses sklearn when available, fallback to numpy."""
    try:
        from sklearn.metrics import pairwise_distances
        return pairwise_distances(a, b, metric="euclidean", n_jobs=-1)
    except ImportError:
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt((diff * diff).sum(axis=-1))


def _knn_radii(features: np.ndarray, k: int) -> np.ndarray:
    """Distance to the k-th nearest neighbour for each row (self excluded)."""
    d = _pairwise(features, features)
    d.sort(axis=1)
    return d[:, k]  # column 0 is self (distance 0)


def compute_prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict[str, float]:
    """Return precision, recall, density, coverage for real vs generated features."""
    real_radii = _knn_radii(real, k)
    fake_radii = _knn_radii(fake, k)
    dist_rf = _pairwise(real, fake)  # (n_real, n_fake)

    # precision: fraction of fakes inside some real k-NN ball
    precision = (dist_rf < real_radii[:, None]).any(axis=0).mean()
    # recall: fraction of reals inside some fake k-NN ball
    recall = (dist_rf < fake_radii[None, :]).any(axis=1).mean()
    # density: avg count of real balls each fake lands in, normalised by k
    density = (1.0 / k) * (dist_rf < real_radii[:, None]).sum(axis=0).mean()
    # coverage: fraction of reals whose nearest fake is within the real k-NN ball
    nearest_fake = dist_rf.min(axis=1)
    coverage = (nearest_fake < real_radii).mean()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
    }
