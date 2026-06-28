"""End-to-end preprocessing report completeness.

Drives ``main()`` over a small synthetic CSV (no real MSD data) and asserts the
outputs exist and the report carries every audit field: skipped rows, invalid
geometries, unmapped labels, and the leakage-checked split summary.
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from shapely.geometry import Polygon

from floorgen.data.preprocess import main


def _sq(x0: float, y0: float, side: float = 3.0) -> str:
    return Polygon(
        [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
    ).wkt


def _row(geom, uid, plan, floor, sub, rt, etype="area") -> dict:
    return {
        "geom": geom, "unit_id": uid, "plan_id": plan, "floor_id": floor,
        "entity_type": etype, "entity_subtype": sub, "roomtype": rt,
    }


def _write_csv(path):
    rows = [
        # unit 1 (plan 100): two adjacent rooms, one label unknown -> unmapped
        _row(_sq(0, 0), 1, 100, 1000, "BEDROOM", "Bedroom"),
        _row(_sq(3, 0), 1, 100, 1000, "MYSTERY", "NotAType"),
        # unit 2 (plan 200): two known rooms
        _row(_sq(10, 0), 2, 200, 2000, "KITCHEN", "Kitchen"),
        _row(_sq(13, 0), 2, 200, 2000, "BATHROOM", "Bathroom"),
        # unit 3 (plan 200): single room -> skipped (<2 rooms)
        _row(_sq(20, 0), 3, 200, 2000, "BEDROOM", "Bedroom"),
        # unit 4: malformed geometry -> dropped + counted as invalid
        _row("NOT WKT AT ALL", 4, 300, 3000, "BEDROOM", "Bedroom"),
        # a non-area row -> filtered out entirely
        _row(_sq(0, 0), 5, 100, 1000, "WALL", "Wall", etype="wall"),
        # a valid area polygon with no unit_id -> dropped from unit grouping, counted
        _row(_sq(50, 0), "", 400, 4000, "BEDROOM", "Bedroom"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_official_split_csvs(train_path, test_path):
    train_rows = [
        _row(_sq(0, 0), 101, 1001, 9001, "BEDROOM", "Bedroom"),
        _row(_sq(3, 0), 101, 1001, 9001, "KITCHEN", "Kitchen"),
        _row(_sq(10, 0), 102, 1002, 9002, "BEDROOM", "Bedroom"),
        _row(_sq(13, 0), 102, 1002, 9002, "BATHROOM", "Bathroom"),
    ]
    test_rows = [
        _row(_sq(20, 0), 201, 2001, 9901, "BEDROOM", "Bedroom"),
        _row(_sq(23, 0), 201, 2001, 9901, "KITCHEN", "Kitchen"),
    ]
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)


def _run(tmp_path, monkeypatch):
    csv = tmp_path / "synthetic.csv"
    out = tmp_path / "processed"
    reports = tmp_path / "reports"
    _write_csv(csv)
    monkeypatch.setattr(sys, "argv", [
        "preprocess", "--csv", str(csv), "--out", str(out), "--reports", str(reports),
    ])
    rc = main()
    report = json.loads((reports / "preprocess_report.json").read_text())
    return rc, out, report


def _run_official(tmp_path, monkeypatch):
    train_csv = tmp_path / "official_train.csv"
    test_csv = tmp_path / "official_test.csv"
    out = tmp_path / "processed"
    reports = tmp_path / "reports"
    _write_official_split_csvs(train_csv, test_csv)
    monkeypatch.setattr(sys, "argv", [
        "preprocess",
        "--train-csv", str(train_csv),
        "--test-csv", str(test_csv),
        "--out", str(out),
        "--reports", str(reports),
        "--val-frac", "0.5",
    ])
    rc = main()
    records = [
        json.loads(line)
        for line in (out / "units.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report = json.loads((reports / "preprocess_report.json").read_text())
    return rc, records, report


def _run_kaggle_dir_floor_split(tmp_path, monkeypatch):
    kaggle_dir = tmp_path / "kaggle"
    csv = kaggle_dir / "mds_V2_5.372k.csv"
    out = tmp_path / "processed"
    reports = tmp_path / "reports"
    kaggle_dir.mkdir()

    rows = [
        _row(_sq(0, 0), 101, 1001, 9001, "BEDROOM", "Bedroom"),
        _row(_sq(3, 0), 101, 1001, 9001, "KITCHEN", "Kitchen"),
        _row(_sq(10, 0), 102, 1002, 9002, "BEDROOM", "Bedroom"),
        _row(_sq(13, 0), 102, 1002, 9002, "BATHROOM", "Bathroom"),
        _row(_sq(20, 0), 201, 2001, 9901, "BEDROOM", "Bedroom"),
        _row(_sq(23, 0), 201, 2001, 9901, "KITCHEN", "Kitchen"),
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)
    for split, floors in {"train": [9001, 9002], "test": [9901]}.items():
        split_dir = kaggle_dir / "modified-swiss-dwellings-v2" / split / "full_out"
        split_dir.mkdir(parents=True)
        for floor in floors:
            (split_dir / f"{floor}.npy").write_bytes(b"marker")

    monkeypatch.setattr(sys, "argv", [
        "preprocess",
        "--kaggle-dir", str(kaggle_dir),
        "--out", str(out),
        "--reports", str(reports),
        "--val-frac", "0.5",
    ])
    rc = main()
    records = [
        json.loads(line)
        for line in (out / "units.jsonl").read_text().splitlines()
        if line.strip()
    ]
    report = json.loads((reports / "preprocess_report.json").read_text())
    return rc, records, report


def test_main_succeeds_and_writes_outputs(tmp_path, monkeypatch):
    rc, out, _ = _run(tmp_path, monkeypatch)
    assert rc == 0
    assert (out / "units.jsonl").exists()
    assert (out / "manifest.parquet").exists()
    lines = (out / "units.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2  # units 1 and 2


def test_report_counts_units_skips_and_invalid(tmp_path, monkeypatch):
    _, _, report = _run(tmp_path, monkeypatch)
    assert report["n_units"] == 2
    assert report["n_skipped"] == 1          # unit 3, single room
    assert report["n_invalid_geom_rows"] == 1  # unit 4, malformed WKT


def test_report_counts_area_rows_without_unit_id(tmp_path, monkeypatch):
    _, _, report = _run(tmp_path, monkeypatch)
    # The blank-unit_id area polygon is dropped from grouping but reported.
    assert report["n_area_rows_no_unit"] == 1


def test_report_flags_unmapped_labels(tmp_path, monkeypatch):
    _, _, report = _run(tmp_path, monkeypatch)
    assert report["n_unmapped_rooms"] == 1
    key = "subtype=MYSTERY|roomtype=NotAType"
    assert report["unmapped_label_sources"].get(key) == 1


def test_report_has_leakage_checked_split_summary(tmp_path, monkeypatch):
    _, _, report = _run(tmp_path, monkeypatch)
    summary = report["split_summary"]
    assert summary["n_plans"] == 2
    assert summary["plan_leakage"] == []
    assert sum(summary["unit_counts"].values()) == 2


def test_report_includes_room_count_distribution(tmp_path, monkeypatch):
    _, _, report = _run(tmp_path, monkeypatch)
    dist = report["room_count_distribution"]
    # JSON serialises histogram keys as strings; both usable units have 2 rooms.
    assert dist["histogram"] == {"2": 2}
    assert dist["suggested_max_rooms_k"] == 2


def test_report_label_frequencies_cover_known_rooms(tmp_path, monkeypatch):
    _, _, report = _run(tmp_path, monkeypatch)
    freqs = report["label_frequencies"]
    assert {"Bedroom", "Structure", "Kitchen", "Bathroom"} <= set(freqs)


def test_official_train_test_mode_preserves_ids_and_test_split(tmp_path, monkeypatch):
    rc, records, report = _run_official(tmp_path, monkeypatch)

    assert rc == 0
    by_unit = {record["unit_id"]: record for record in records}
    assert by_unit[201]["split"] == "test"
    assert by_unit[201]["official_split"] == "test"
    assert by_unit[201]["plan_id"] == 2001
    assert by_unit[201]["floor_id"] == 9901
    assert {by_unit[101]["split"], by_unit[102]["split"]} == {"train", "val"}
    assert report["split_counts"]["test"] == 1
    assert report["official_split_counts"] == {"train": 2, "test": 1}


def test_kaggle_dir_mode_uses_floor_id_split_markers(tmp_path, monkeypatch):
    rc, records, report = _run_kaggle_dir_floor_split(tmp_path, monkeypatch)

    assert rc == 0
    by_unit = {record["unit_id"]: record for record in records}
    assert by_unit[201]["split"] == "test"
    assert by_unit[201]["official_split"] == "test"
    assert by_unit[201]["official_split_source"] == "kaggle_dir_floor_id"
    assert {by_unit[101]["split"], by_unit[102]["split"]} == {"train", "val"}
    assert report["split_counts"]["test"] == 1
    assert report["official_split_counts"] == {"train": 2, "test": 1}
    assert report["official_split_source_counts"] == {"kaggle_dir_floor_id": 3}
