"""MSD preprocessing pipeline.

Turns the raw geometry CSV into model-ready per-apartment records plus a
leakage-safe train/val split and a human-readable report.

One ``unit_id`` == one apartment == one training example. We group the split by
``plan_id``/``floor_id`` so apartments from the same physical floor never
straddle the train/val boundary.

Run:
    MSD_CSV_PATH=/path/to/mds_V2_5.372k.csv python -m floorgen.data.preprocess \
        --out data/processed --limit-units 0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import PATHS, ROOM_NAME_TO_IDX, SEED
from ..seeding import seed_everything
from .normalize import fit_transform, scale_features
from .outline import build_outline, canonical_room_name

REQUIRED_COLS = {
    "geom", "unit_id", "plan_id", "floor_id",
    "entity_type", "entity_subtype", "roomtype",
}


def _load(csv_path: Path) -> Any:
    import geopandas as gpd
    import pandas as pd
    from shapely import wkt

    if not csv_path.exists():
        raise FileNotFoundError(f"MSD CSV not found: {csv_path}. Set MSD_CSV_PATH.")
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df[df["entity_type"] == "area"].copy()
    df["geom"] = df["geom"].apply(wkt.loads)
    return gpd.GeoDataFrame(df, geometry="geom")


def build_records(gdf: Any, limit_units: int = 0) -> tuple[list[dict], int]:
    """One record per unit_id: outline, rooms, transform, scale features, meta."""
    records: list[dict] = []
    unit_ids = list(gdf["unit_id"].dropna().unique())
    if limit_units:
        unit_ids = unit_ids[:limit_units]

    skipped = 0
    for uid in unit_ids:
        u = gdf[gdf["unit_id"] == uid]
        room_geoms = list(u.geometry)
        if len(room_geoms) < 2:
            skipped += 1
            continue
        try:
            outline = build_outline(room_geoms)
        except Exception:
            skipped += 1
            continue
        if outline.is_empty or outline.area <= 0:
            skipped += 1
            continue

        tf = fit_transform(outline)
        feats = scale_features(outline)

        rooms = []
        for _, row in u.iterrows():
            label = canonical_room_name(row.get("entity_subtype"), row.get("roomtype"))
            geom = row["geom"]
            rooms.append({
                "label": label,
                "label_idx": ROOM_NAME_TO_IDX[label],
                "wkt": geom.wkt,
                "area_m2": float(geom.area),
            })

        records.append({
            "unit_id": int(uid),
            "plan_id": int(u["plan_id"].iloc[0]) if u["plan_id"].notna().any() else -1,
            "floor_id": int(u["floor_id"].iloc[0]) if u["floor_id"].notna().any() else -1,
            "n_rooms": len(rooms),
            "outline_wkt": outline.wkt,
            "transform": asdict(tf),
            "scale_features": asdict(feats),
            "rooms": rooms,
        })
    return records, skipped


def split_by_plan(records: list[dict], val_frac: float = 0.15, seed: int = SEED) -> dict[int, str]:
    """Assign each unit to train/val, grouping by plan_id (no floor leakage)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    plan_ids = sorted({r["plan_id"] for r in records})
    rng.shuffle(plan_ids)
    n_val = max(1, int(len(plan_ids) * val_frac))
    val_plans = set(plan_ids[:n_val])
    return {r["unit_id"]: ("val" if r["plan_id"] in val_plans else "train") for r in records}


def write_outputs(records: list[dict], split: dict[int, str], out_dir: Path,
                  reports_dir: Path, skipped: int) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        r["split"] = split[r["unit_id"]]

    jsonl_path = out_dir / "units.jsonl"
    with jsonl_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    manifest_rows = []
    label_counter: Counter = Counter()
    for r in records:
        lab_hist = Counter(rm["label"] for rm in r["rooms"])
        label_counter.update(lab_hist)
        manifest_rows.append({
            "unit_id": r["unit_id"],
            "plan_id": r["plan_id"],
            "floor_id": r["floor_id"],
            "split": r["split"],
            "n_rooms": r["n_rooms"],
            "area_m2": r["scale_features"]["area_m2"],
            "bbox_w_m": r["scale_features"]["bbox_w_m"],
            "bbox_h_m": r["scale_features"]["bbox_h_m"],
            **{f"n_{k}": v for k, v in lab_hist.items()},
        })
    manifest = pd.DataFrame(manifest_rows).fillna(0)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)

    report = {
        "n_units": len(records),
        "n_skipped": skipped,
        "n_plans": int(manifest["plan_id"].nunique()),
        "n_floors": int(manifest["floor_id"].nunique()),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "rooms_per_unit": {
            "min": int(manifest["n_rooms"].min()),
            "median": float(manifest["n_rooms"].median()),
            "mean": float(manifest["n_rooms"].mean()),
            "max": int(manifest["n_rooms"].max()),
        },
        "area_m2": {
            "min": float(manifest["area_m2"].min()),
            "median": float(manifest["area_m2"].median()),
            "max": float(manifest["area_m2"].max()),
        },
        "label_frequencies": dict(label_counter.most_common()),
    }
    (reports_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess MSD into model-ready unit records.")
    p.add_argument("--csv", type=Path, default=PATHS.msd_csv)
    p.add_argument("--out", type=Path, default=PATHS.processed_dir)
    p.add_argument("--reports", type=Path, default=PATHS.reports_dir)
    p.add_argument("--limit-units", type=int, default=0, help="0 = all units.")
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(SEED)
    gdf = _load(args.csv)
    records, skipped = build_records(gdf, limit_units=args.limit_units)
    if not records:
        print("error: no usable units produced.")
        return 1
    split = split_by_plan(records, val_frac=args.val_frac, seed=SEED)
    write_outputs(records, split, args.out, args.reports, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
