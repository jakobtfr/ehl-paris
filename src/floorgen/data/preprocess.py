"""MSD preprocessing pipeline.

Turns the raw geometry CSV into model-ready per-apartment records plus a
human-readable report.

For official submission runs, pass Kaggle's predefined train/test CSV files.
The official test rows remain ``split == "test"`` and only the official train
rows are divided into local train/val. ``unit_id``, ``plan_id``, and
``floor_id`` are copied through unchanged so generated outputs can be matched
back to the challenge split.

For local development, a single CSV can still be split into train/val by
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
from .outline import build_outline, classify_room

REQUIRED_COLS = {
    "geom", "unit_id", "plan_id", "floor_id",
    "entity_type", "entity_subtype", "roomtype",
}


def _try_load_wkt(value: Any) -> Any:
    """Parse one WKT cell, returning None for anything unusable.

    A non-string (NaN/None), unparseable WKT, or empty geometry yields None so a
    single bad row never aborts the whole load. Real parse failures are surfaced
    as a count by :func:`parse_area_geometries`, not silently swallowed.
    """
    from shapely import wkt

    if not isinstance(value, str):
        return None
    try:
        g = wkt.loads(value)
    except Exception:
        return None
    return None if (g is None or g.is_empty) else g


def parse_area_geometries(df: Any) -> tuple[Any, int]:
    """Validate schema, keep ``entity_type == "area"`` rows, parse geometries.

    Returns ``(gdf, n_invalid)`` where ``n_invalid`` counts area rows whose
    geometry could not be parsed (or was empty) and were dropped.
    """
    import geopandas as gpd

    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    area = df[df["entity_type"] == "area"].copy()
    parsed = area["geom"].apply(_try_load_wkt)
    valid = parsed.notna()
    n_invalid = int((~valid).sum())
    area = area.loc[valid].copy()
    area["geom"] = parsed[valid]
    return gpd.GeoDataFrame(area, geometry="geom"), n_invalid


def _load(csv_path: Path) -> tuple[Any, int]:
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"MSD CSV not found: {csv_path}. Set MSD_CSV_PATH.")
    df = pd.read_csv(csv_path)
    return parse_area_geometries(df)


def _find_split_csv(kaggle_dir: Path, split: str) -> Path:
    """Find one CSV whose filename names the requested Kaggle split."""
    def names_split(path: Path) -> bool:
        tokens = path.stem.lower().replace("-", "_").split("_")
        return any(token == split.lower() or token.startswith(split.lower()) for token in tokens)

    matches = [
        p for p in kaggle_dir.rglob("*.csv")
        if names_split(p)
    ]
    if not matches:
        raise FileNotFoundError(f"no {split!r} CSV found under {kaggle_dir}")
    if len(matches) > 1:
        formatted = ", ".join(str(p) for p in sorted(matches))
        raise ValueError(f"multiple {split!r} CSV candidates under {kaggle_dir}: {formatted}")
    return matches[0]


def _find_optional_split_csv(kaggle_dir: Path, split: str) -> Path | None:
    try:
        return _find_split_csv(kaggle_dir, split)
    except FileNotFoundError:
        return None


def _find_geometry_csv(kaggle_dir: Path) -> Path:
    """Find the single geometry CSV in an extracted Kaggle archive.

    The public archive ships one large ``mds_*.csv`` with vector geometry and
    split membership in sibling ``train``/``test`` folders, not necessarily as
    separate train/test CSVs.
    """
    matches = [
        p for p in kaggle_dir.rglob("*.csv")
        if "train" not in p.stem.lower() and "test" not in p.stem.lower()
    ]
    if not matches:
        raise FileNotFoundError(f"no geometry CSV found under {kaggle_dir}")
    if len(matches) > 1:
        formatted = ", ".join(str(p) for p in sorted(matches))
        raise ValueError(f"multiple geometry CSV candidates under {kaggle_dir}: {formatted}")
    return matches[0]


def _split_marker_stems(kaggle_dir: Path, split: str) -> set[str]:
    """Load floor-id stems from Kaggle split marker files.

    The MSD archive repeats the same ids under ``struct_in``, ``graph_in``,
    ``graph_out``, and ``full_out``. Prefer ``full_out`` so each floor appears
    once, but fall back to all files below the split directory for robustness.
    """
    split_dir = kaggle_dir / "modified-swiss-dwellings-v2" / split
    full_out = split_dir / "full_out"
    search_dir = full_out if full_out.exists() else split_dir
    if not search_dir.exists():
        raise FileNotFoundError(f"no Kaggle {split!r} split marker directory at {search_dir}")
    stems = {p.stem for p in search_dir.rglob("*") if p.is_file()}
    if not stems:
        raise FileNotFoundError(f"no Kaggle {split!r} split marker files under {search_dir}")
    return stems


def _load_kaggle_floor_splits(kaggle_dir: Path) -> dict[int, str]:
    """Return ``floor_id -> official_split`` from Kaggle split folders."""
    result: dict[int, str] = {}
    for split in ("train", "test"):
        for stem in _split_marker_stems(kaggle_dir, split):
            try:
                floor_id = int(stem)
            except ValueError as exc:
                raise ValueError(
                    f"Kaggle {split!r} marker file has non-integer floor id stem: {stem!r}"
                ) from exc
            previous = result.get(floor_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"floor_id={floor_id} appears in both official {previous!r} and {split!r}"
                )
            result[floor_id] = split
    return result


def apply_official_split_by_floor_id(
    records: list[dict],
    floor_splits: dict[int, str],
) -> None:
    """Annotate records with official split membership from ``floor_id`` markers."""
    missing = sorted({int(record["floor_id"]) for record in records} - set(floor_splits))
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} processed floor_id values are missing from Kaggle split markers: "
            f"{preview}{suffix}"
        )
    for record in records:
        record["official_split"] = floor_splits[int(record["floor_id"])]
        record["official_split_source"] = "kaggle_dir_floor_id"


def build_records(gdf: Any, limit_units: int = 0) -> tuple[list[dict], int, Counter]:
    """One record per unit_id: outline, rooms, transform, scale features, meta.

    Returns ``(records, skipped, unmapped)`` where ``unmapped`` counts the raw
    ``subtype=...|roomtype=...`` sources that fell back to 'Structure' because
    they were not recognised, so unknown labels stay auditable.
    """
    records: list[dict] = []
    unmapped: Counter = Counter()
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
            subtype, roomtype = row.get("entity_subtype"), row.get("roomtype")
            label, matched = classify_room(subtype, roomtype)
            if not matched:
                unmapped[f"subtype={subtype}|roomtype={roomtype}"] += 1
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
    return records, skipped, unmapped


def build_records_from_csv(
    csv_path: Path,
    *,
    official_split: str | None = None,
    limit_units: int = 0,
) -> tuple[list[dict], int, Counter, int, int]:
    """Load one MSD CSV and return processed records plus audit counts."""
    gdf, n_invalid_geom = _load(csv_path)
    n_area_rows_no_unit = int(gdf["unit_id"].isna().sum())
    records, skipped, unmapped = build_records(gdf, limit_units=limit_units)
    if official_split is not None:
        for record in records:
            record["official_split"] = official_split
    return records, skipped, unmapped, n_invalid_geom, n_area_rows_no_unit


def split_by_plan(records: list[dict], val_frac: float = 0.15, seed: int = SEED) -> dict[int, str]:
    """Assign each unit to train/val, grouping by plan_id (no floor leakage)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    plan_ids = sorted({r["plan_id"] for r in records})
    rng.shuffle(plan_ids)
    n_val = max(1, int(len(plan_ids) * val_frac))
    val_plans = set(plan_ids[:n_val])
    return {r["unit_id"]: ("val" if r["plan_id"] in val_plans else "train") for r in records}


def split_predefined_train_test(
    records: list[dict],
    val_frac: float = 0.15,
    seed: int = SEED,
) -> dict[int, str]:
    """Respect Kaggle's train/test split and derive val only from train rows."""
    seen: dict[int, str] = {}
    train_records: list[dict] = []
    split: dict[int, str] = {}

    for record in records:
        unit_id = int(record["unit_id"])
        source = str(record.get("official_split", "")).lower()
        if source not in {"train", "test"}:
            raise ValueError(
                f"record unit_id={unit_id} has unsupported official_split={source!r}"
            )
        previous = seen.get(unit_id)
        if previous is not None and previous != source:
            raise ValueError(
                f"unit_id={unit_id} appears in both official {previous!r} and {source!r}"
            )
        seen[unit_id] = source

        if source == "test":
            split[unit_id] = "test"
        else:
            train_records.append(record)

    if not train_records:
        raise ValueError("official train split produced no usable records")

    split.update(split_by_plan(train_records, val_frac=val_frac, seed=seed))
    return split


def room_count_distribution(records: list[dict]) -> dict[str, Any]:
    """Histogram + percentiles of rooms-per-unit, with a suggested slot count.

    MAX_ROOMS_K (the model's fixed slot count) should be chosen from real data;
    ``suggested_max_rooms_k`` is the 99th percentile rounded up, i.e. the
    smallest slot count that fits ~99% of units. Reported so the choice is
    data-driven and auditable.
    """
    import numpy as np

    counts = [int(r["n_rooms"]) for r in records]
    if not counts:
        return {"histogram": {}, "p50": 0, "p90": 0, "p95": 0, "p99": 0,
                "suggested_max_rooms_k": 0}
    hist = Counter(counts)
    pcts = {f"p{q}": float(np.percentile(counts, q)) for q in (50, 90, 95, 99)}
    return {
        "histogram": {int(k): int(v) for k, v in sorted(hist.items())},
        **pcts,
        "suggested_max_rooms_k": int(np.ceil(pcts["p99"])),
    }


def split_summary(records: list[dict], split: dict[int, str]) -> dict[str, Any]:
    """Unit- and plan-level split counts plus an explicit leakage check.

    The split is leakage-safe when every unit of a plan shares one split.
    ``plan_leakage`` lists any ``plan_id`` whose units landed in more than one
    split (should always be empty); it makes the guarantee auditable rather than
    merely assumed.
    """
    unit_counts: Counter = Counter(split[r["unit_id"]] for r in records)
    plan_to_splits: dict[int, set[str]] = {}
    for r in records:
        plan_id, s = r["plan_id"], split[r["unit_id"]]
        plan_to_splits.setdefault(plan_id, set()).add(s)
    leaked = {plan_id for plan_id, splits in plan_to_splits.items() if len(splits) > 1}
    plan_counts: Counter = Counter(
        next(iter(splits)) if len(splits) == 1 else "mixed"
        for splits in plan_to_splits.values()
    )
    return {
        "unit_counts": dict(unit_counts),
        "plan_counts": dict(plan_counts),
        "n_plans": len(plan_to_splits),
        "plan_leakage": sorted(leaked),
    }


def write_outputs(records: list[dict], split: dict[int, str], out_dir: Path,
                  reports_dir: Path, skipped: int,
                  unmapped: Counter | None = None,
                  n_invalid_geom: int = 0,
                  n_area_rows_no_unit: int = 0) -> None:
    import pandas as pd

    unmapped = unmapped or Counter()

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
        "n_invalid_geom_rows": int(n_invalid_geom),
        "n_area_rows_no_unit": int(n_area_rows_no_unit),
        "n_plans": int(manifest["plan_id"].nunique()),
        "n_floors": int(manifest["floor_id"].nunique()),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "official_split_counts": dict(
            Counter(str(r.get("official_split", "derived")) for r in records)
        ),
        "official_split_source_counts": dict(
            Counter(str(r.get("official_split_source", "none")) for r in records)
        ),
        "split_summary": split_summary(records, split),
        "room_count_distribution": room_count_distribution(records),
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
        "n_unmapped_rooms": int(sum(unmapped.values())),
        "unmapped_label_sources": dict(unmapped.most_common(20)),
    }
    (reports_dir / "preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess MSD into model-ready unit records.")
    p.add_argument("--csv", type=Path, default=PATHS.msd_csv)
    p.add_argument(
        "--train-csv",
        type=Path,
        default=PATHS.msd_train_csv,
        help="Official Kaggle train split CSV.",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=PATHS.msd_test_csv,
        help="Official Kaggle test split CSV.",
    )
    p.add_argument(
        "--kaggle-dir",
        type=Path,
        default=PATHS.msd_kaggle_dir,
        help="Directory containing one train CSV and one test CSV; used if split paths are omitted.",
    )
    p.add_argument("--out", type=Path, default=PATHS.processed_dir)
    p.add_argument("--reports", type=Path, default=PATHS.reports_dir)
    p.add_argument("--limit-units", type=int, default=0, help="0 = all units.")
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(SEED)
    train_csv = args.train_csv
    test_csv = args.test_csv
    floor_splits: dict[int, str] | None = None
    if args.kaggle_dir:
        if train_csv is None and test_csv is None:
            train_csv = _find_optional_split_csv(args.kaggle_dir, "train")
            test_csv = _find_optional_split_csv(args.kaggle_dir, "test")
            if train_csv is None and test_csv is None:
                floor_splits = _load_kaggle_floor_splits(args.kaggle_dir)
                if not args.csv.exists():
                    args.csv = _find_geometry_csv(args.kaggle_dir)
            elif bool(train_csv) != bool(test_csv):
                raise ValueError(
                    "Kaggle directory contains only one split CSV; provide both "
                    "--train-csv and --test-csv, or use floor-id split folders with one CSV"
                )
        else:
            train_csv = train_csv or _find_split_csv(args.kaggle_dir, "train")
            test_csv = test_csv or _find_split_csv(args.kaggle_dir, "test")

    if bool(train_csv) != bool(test_csv):
        raise ValueError("pass both --train-csv and --test-csv, or neither")

    if train_csv and test_csv:
        train_records, train_skipped, train_unmapped, train_invalid, train_no_unit = (
            build_records_from_csv(train_csv, official_split="train",
                                   limit_units=args.limit_units)
        )
        test_records, test_skipped, test_unmapped, test_invalid, test_no_unit = (
            build_records_from_csv(test_csv, official_split="test",
                                   limit_units=args.limit_units)
        )
        records = train_records + test_records
        skipped = train_skipped + test_skipped
        unmapped = train_unmapped + test_unmapped
        n_invalid_geom = train_invalid + test_invalid
        n_area_rows_no_unit = train_no_unit + test_no_unit
        split = split_predefined_train_test(records, val_frac=args.val_frac, seed=SEED)
    elif floor_splits is not None:
        records, skipped, unmapped, n_invalid_geom, n_area_rows_no_unit = (
            build_records_from_csv(args.csv, limit_units=args.limit_units)
        )
        apply_official_split_by_floor_id(records, floor_splits)
        split = split_predefined_train_test(records, val_frac=args.val_frac, seed=SEED)
    else:
        records, skipped, unmapped, n_invalid_geom, n_area_rows_no_unit = (
            build_records_from_csv(args.csv, limit_units=args.limit_units)
        )
        split = split_by_plan(records, val_frac=args.val_frac, seed=SEED)

    if not records:
        print("error: no usable units produced.")
        return 1
    write_outputs(records, split, args.out, args.reports, skipped, unmapped,
                  n_invalid_geom=n_invalid_geom,
                  n_area_rows_no_unit=n_area_rows_no_unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
