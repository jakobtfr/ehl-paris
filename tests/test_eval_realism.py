from __future__ import annotations

import numpy as np

from floorgen.eval import realism


def test_image_distribution_report_threads_fid_and_prdc(monkeypatch) -> None:
    real_images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    fake_images = np.ones((4, 8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(realism, "compute_fid", lambda _real, _fake: 12.5)
    monkeypatch.setattr(
        realism,
        "inception_features",
        lambda images: np.arange(len(images) * 4, dtype=np.float32).reshape(len(images), 4),
    )
    monkeypatch.setattr(
        realism,
        "compute_prdc",
        lambda _real, _fake, k: {
            "precision": 0.1,
            "recall": 0.2,
            "density": 0.3,
            "coverage": 0.4,
        },
    )

    report = realism.image_distribution_report(real_images, fake_images, prdc_k=5)

    assert report["status"] == "ok"
    assert report["fid"] == 12.5
    assert report["prdc_k"] == 2
    assert report["density"] == 0.3
    assert report["coverage"] == 0.4


def test_try_image_distribution_report_reports_dependency_blocker(monkeypatch) -> None:
    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)

    def missing_dependency(_real, _fake):
        raise ImportError("torchmetrics missing")

    monkeypatch.setattr(realism, "compute_fid", missing_dependency)

    report = realism.try_image_distribution_report(images, images)

    assert report["status"] == "blocked"
    assert "torchmetrics missing" in report["error"]
    assert report["fid"] is None
