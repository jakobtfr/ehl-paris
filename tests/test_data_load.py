"""CSV loading + invalid-geometry handling tests.

A single malformed WKT cell must not abort preprocessing of the whole dataset:
unparseable / empty geometries are dropped and counted, and schema problems
raise clearly. Synthetic CSV/data only — no real MSD file is read.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import Polygon

from floorgen.data.preprocess import REQUIRED_COLS, _load, parse_area_geometries

_SQUARE_WKT = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]).wkt


def _raw_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=sorted(REQUIRED_COLS))


def _row(geom, entity_type="area", **kw) -> dict:
    base = {
        "geom": geom, "unit_id": 1, "plan_id": 1, "floor_id": 1,
        "entity_type": entity_type, "entity_subtype": "BEDROOM", "roomtype": "Bedroom",
    }
    base.update(kw)
    return base


def test_parse_area_keeps_only_area_rows():
    df = _raw_df([
        _row(_SQUARE_WKT, entity_type="area"),
        _row(_SQUARE_WKT, entity_type="wall"),
        _row(_SQUARE_WKT, entity_type="opening"),
    ])
    gdf, n_invalid = parse_area_geometries(df)
    assert len(gdf) == 1
    assert n_invalid == 0
    assert set(gdf["entity_type"]) == {"area"}


def test_parse_area_drops_malformed_wkt_and_counts_it():
    df = _raw_df([
        _row(_SQUARE_WKT),
        _row("THIS IS NOT WKT"),
        _row(_SQUARE_WKT),
    ])
    gdf, n_invalid = parse_area_geometries(df)
    assert len(gdf) == 2
    assert n_invalid == 1
    assert all(g.geom_type == "Polygon" for g in gdf.geometry)


def test_parse_area_drops_nan_geometry():
    df = _raw_df([_row(_SQUARE_WKT), _row(float("nan"))])
    gdf, n_invalid = parse_area_geometries(df)
    assert len(gdf) == 1
    assert n_invalid == 1


def test_parse_area_drops_empty_geometry():
    df = _raw_df([_row(_SQUARE_WKT), _row("POLYGON EMPTY")])
    gdf, n_invalid = parse_area_geometries(df)
    assert len(gdf) == 1
    assert n_invalid == 1


def test_parse_area_missing_column_raises():
    df = pd.DataFrame({"geom": [_SQUARE_WKT], "unit_id": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        parse_area_geometries(df)


def test_load_from_csv_tolerates_bad_row(tmp_path):
    df = _raw_df([
        _row(_SQUARE_WKT, unit_id=1),
        _row("garbage", unit_id=2),
        _row(_SQUARE_WKT, unit_id=3, entity_type="wall"),
    ])
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    gdf, n_invalid = _load(csv)
    assert n_invalid == 1            # the garbage WKT
    assert len(gdf) == 1            # unit 1 only (unit 3 is a wall, filtered)
    assert int(gdf["unit_id"].iloc[0]) == 1


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load(tmp_path / "nope.csv")
