"""Smoke test: verify the complete generation pipeline works end-to-end.

Run without external data, GPU, or gradio. In this submission workspace it uses
the default AMD checkpoint, so install/run with the training extra for torch.
Exercises:
  - generate() contract (outline → room records)
  - sample_layouts() with multiple samples
  - validity metrics (geometry health check)
  - determinism (same seed → same output)
  - export pipeline (room records → GeoJSON)

Usage:
    uv run --extra train python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shapely.geometry import Polygon, box

from floorgen.config import ROOM_NAMES, SEED
from floorgen.eval.metrics import validity_metrics
from floorgen.generate import backend_provenance, generate, sample_layouts


def main() -> int:
    print("=" * 60)
    print("floorgen smoke test")
    print("=" * 60)

    # 1. Basic generation
    print("\n[1/5] generate(box(0,0,10,8))...")
    outline = box(0, 0, 10, 8)
    layout = generate(outline)
    assert len(layout) >= 1, "Expected at least 1 room"
    for r in layout:
        assert r["label"] in ROOM_NAMES
        assert r["polygon"].is_valid
    print(f"  OK: {len(layout)} rooms, all valid polygons")

    # 2. L-shaped outline
    print("\n[2/5] generate(L-shaped outline)...")
    l_shape = Polygon([(0, 0), (10, 0), (10, 6), (6, 6), (6, 10), (0, 10)])
    layout2 = generate(l_shape)
    assert len(layout2) >= 1
    print(f"  OK: {len(layout2)} rooms")

    # 3. Validity metrics
    print("\n[3/5] Validity metrics...")
    parts = [(r["polygon"], r["label_idx"]) for r in layout]
    metrics = validity_metrics(parts, outline)
    print(f"  outside_frac: {metrics['outside_frac']:.4f}")
    print(f"  overlap_frac: {metrics['overlap_frac']:.4f}")
    print(f"  gap_frac:     {metrics['gap_frac']:.4f}")
    print(f"  invalid_rate: {metrics['invalid_rate']:.4f}")
    assert metrics["outside_frac"] < 0.02, "Too much area outside outline"
    assert metrics["invalid_rate"] == 0.0, "Invalid polygons found"
    print("  OK: geometry health within tolerance")

    # 4. Determinism
    print("\n[4/5] Determinism check...")
    a = generate(outline)
    b = generate(outline)
    assert len(a) == len(b), "Non-deterministic room count"
    for ra, rb in zip(a, b):
        assert ra["label_idx"] == rb["label_idx"], "Non-deterministic labels"
        diff = ra["polygon"].symmetric_difference(rb["polygon"]).area
        assert diff < 1e-9, f"Non-deterministic geometry: diff={diff}"
    print("  OK: same seed -> identical output")

    # 5. Multi-sample + GeoJSON export
    print("\n[5/5] Multi-sample + GeoJSON export...")
    samples = sample_layouts(outline, seed=SEED, n_samples=3)
    assert len(samples) == 3
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": r["label"]},
                "geometry": r["geojson"],
            }
            for r in samples[0]
        ],
    }
    serialized = json.dumps(geojson)
    assert len(serialized) > 100
    print(f"  OK: 3 samples generated, GeoJSON serializes ({len(serialized)} chars)")

    # Summary
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print(f"  Backend: {backend_provenance()}")
    print(f"  Seed: {SEED}")
    print(f"  Room types: {', '.join(ROOM_NAMES)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
