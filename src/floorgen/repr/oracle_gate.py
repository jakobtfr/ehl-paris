"""Oracle reconstruction gate.

Before training any model, prove the MRR representation can reconstruct real
floor plans. We encode each real room to its minimum rotated rectangle, run ONLY
the deterministic repair layer, and compare the result to the original rooms.

If MRRs reconstruct real plans poorly, no generative model on top of MRRs can
score well -- only then use axis-aligned boxes as a debug fallback or escalate to
corner sequences. This is the plan's hours 6-10 go/no-go, made measurable.

Metrics (per unit, averaged):
  - per-room IoU between original room and its reconstructed cell (matched by
    label + centroid),
  - room retention: reconstructed cells / original rooms,
  - total-area error: |sum reconstructed - outline area| / outline area,
  - outside-outline area fraction (should be ~0 after repair).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from shapely import wkt
from shapely.geometry import Polygon

from ..config import PATHS
from .mrr import RepairRejected, polygon_to_mrr, repair_partition


def _match_iou(originals: list[tuple[Polygon, int]],
               recon: list[tuple[Polygon, int]]) -> list[float]:
    """Greedy match recon cells to originals by same label + centroid distance,
    return IoU per matched original."""
    ious: list[float] = []
    used = set()
    for o_poly, o_label in originals:
        oc = o_poly.centroid
        best_j, best_d = -1, float("inf")
        for j, (r_poly, r_label) in enumerate(recon):
            if j in used or r_label != o_label:
                continue
            d = r_poly.centroid.distance(oc)
            if d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used.add(best_j)
            r_poly = recon[best_j][0]
            inter = o_poly.intersection(r_poly).area
            union = o_poly.union(r_poly).area
            ious.append(inter / union if union > 0 else 0.0)
        else:
            ious.append(0.0)  # original room with no reconstructed match
    return ious


def evaluate_units(units: list[dict]) -> dict:
    per_unit = []
    repair_failures = 0
    for u in units:
        outline = wkt.loads(u["outline_wkt"])
        originals = [(wkt.loads(rm["wkt"]), rm["label_idx"]) for rm in u["rooms"]]
        mrrs = [polygon_to_mrr(p, lab) for p, lab in originals]
        repair_failed = False
        try:
            recon = repair_partition(mrrs, outline)
        except RepairRejected:
            repair_failed = True
            repair_failures += 1
            recon = []

        ious = _match_iou(originals, recon)
        recon_area = sum(p.area for p, _ in recon)
        outside = sum(max(p.difference(outline).area, 0.0) for p, _ in recon)
        per_unit.append({
            "unit_id": u["unit_id"],
            "n_orig": len(originals),
            "n_recon": len(recon),
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "retention": len(recon) / len(originals) if originals else 0.0,
            "area_err": abs(recon_area - outline.area) / outline.area if outline.area else 0.0,
            "outside_frac": outside / outline.area if outline.area else 0.0,
            "repair_failed": repair_failed,
        })

    def agg(key: str) -> float:
        return float(np.mean([p[key] for p in per_unit])) if per_unit else 0.0

    return {
        "n_units": len(per_unit),
        "mean_iou": agg("mean_iou"),
        "median_iou": float(np.median([p["mean_iou"] for p in per_unit])) if per_unit else 0.0,
        "mean_retention": agg("retention"),
        "mean_area_err": agg("area_err"),
        "mean_outside_frac": agg("outside_frac"),
        "repair_failure_rate": repair_failures / len(per_unit) if per_unit else 0.0,
        "per_unit": per_unit,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run the oracle MRR-reconstruction gate.")
    p.add_argument("--units", type=Path,
                   default=PATHS.processed_dir / "units.jsonl")
    p.add_argument("--reports", type=Path, default=PATHS.reports_dir)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    units = []
    with args.units.open() as f:
        for line in f:
            units.append(json.loads(line))
            if args.limit and len(units) >= args.limit:
                break

    result = evaluate_units(units)
    args.reports.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_unit"}
    (args.reports / "oracle_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2))

    # Advisory verdict
    ok = (
        result["mean_iou"] >= 0.65
        and result["mean_outside_frac"] < 0.01
        and result["repair_failure_rate"] < 0.05
    )
    print(f"\nVERDICT: {'PASS' if ok else 'REVIEW'} "
          f"(MRRs {'reconstruct real plans acceptably' if ok else 'may need fallback representation review'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
