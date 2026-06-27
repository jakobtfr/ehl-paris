"""Label-mapping audit tests.

Unknown MSD room labels must never be silently invented: they fall back to the
``Structure`` class *and* are reported, so data drift (a new subtype, a typo)
surfaces in the preprocessing report instead of vanishing into Structure.

All geometry here is synthetic — no real MSD data is read.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from floorgen.data.outline import canonical_room_name, classify_room
from floorgen.data.preprocess import build_records


def _square(x0: float, y0: float, side: float = 3.0) -> Polygon:
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


def _gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame shaped exactly like preprocess._load output."""
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry="geom")


# --- classify_room ---------------------------------------------------------

def test_classify_known_subtype_is_matched():
    label, matched = classify_room("BEDROOM", None)
    assert label == "Bedroom"
    assert matched is True


def test_classify_subtype_case_insensitive():
    assert classify_room("  bedroom  ", None) == ("Bedroom", True)


def test_classify_falls_back_to_roomtype_when_subtype_unknown():
    label, matched = classify_room("TOTALLY_UNKNOWN", "Kitchen")
    assert label == "Kitchen"
    assert matched is True


def test_classify_unknown_is_structure_and_flagged():
    label, matched = classify_room("MYSTERY_ROOM", "NotARealType")
    assert label == "Structure"
    assert matched is False


def test_classify_none_inputs_are_unmatched_structure():
    assert classify_room(None, None) == ("Structure", False)


def test_canonical_room_name_agrees_with_classify():
    for sub, rt in [("BEDROOM", None), ("FOO", "Kitchen"), ("FOO", "Bar"), (None, None)]:
        assert canonical_room_name(sub, rt) == classify_room(sub, rt)[0]


# --- build_records unmapped reporting --------------------------------------

def test_build_records_reports_unmapped_label_sources():
    rows = [
        # unit 1: one known room, one genuinely unknown room
        {"unit_id": 1, "plan_id": 10, "floor_id": 100,
         "entity_subtype": "BEDROOM", "roomtype": "Bedroom", "geom": _square(0, 0)},
        {"unit_id": 1, "plan_id": 10, "floor_id": 100,
         "entity_subtype": "MYSTERY_ROOM", "roomtype": "NotARealType", "geom": _square(3, 0)},
    ]
    records, skipped, unmapped = build_records(_gdf(rows))

    assert skipped == 0
    assert len(records) == 1
    labels = [rm["label"] for rm in records[0]["rooms"]]
    assert labels == ["Bedroom", "Structure"]
    # The unknown row is counted once, keyed by its raw subtype/roomtype.
    assert sum(unmapped.values()) == 1
    assert unmapped["subtype=MYSTERY_ROOM|roomtype=NotARealType"] == 1


def test_build_records_no_unmapped_when_all_known():
    rows = [
        {"unit_id": 7, "plan_id": 1, "floor_id": 1,
         "entity_subtype": "KITCHEN", "roomtype": "Kitchen", "geom": _square(0, 0)},
        {"unit_id": 7, "plan_id": 1, "floor_id": 1,
         "entity_subtype": "BATHROOM", "roomtype": "Bathroom", "geom": _square(3, 0)},
    ]
    records, skipped, unmapped = build_records(_gdf(rows))
    assert skipped == 0
    assert sum(unmapped.values()) == 0
