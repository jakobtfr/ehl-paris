from __future__ import annotations

import json
import os

from shapely.geometry import box

from floorgen.demo.evidence import (
    EXPORT_COLUMNS,
    build_export_rows,
    candidate_budget_from,
    generation_mode_code,
    layout_to_geojson,
    load_metric_status,
)


def _room(label: str = "Bedroom", label_idx: int = 0) -> dict:
    poly = box(0, 0, 2, 3)
    return {
        "label": label,
        "label_idx": label_idx,
        "polygon": poly,
        "geojson": poly.__geo_interface__,
    }


def test_generation_mode_and_candidate_budget_helpers() -> None:
    assert generation_mode_code("Ranked/post-processed") == "ranked"
    assert generation_mode_code("Raw samples") == "raw"
    assert candidate_budget_from("4") == 4
    assert candidate_budget_from(0) == 1
    assert candidate_budget_from(None, default=12) == 12


def test_geojson_and_export_rows_use_vector_schema() -> None:
    layout = [_room("Kitchen", 2)]

    geojson = layout_to_geojson(layout)
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"][0]["properties"]["label"] == "Kitchen"
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"

    rows = build_export_rows([
        {
            "unit_id": "unit-1",
            "input": "Selected",
            "sample_idx": 0,
            "seed": 42,
            "mode": "ranked",
            "layout": layout,
        }
    ])
    assert list(rows[0]) == EXPORT_COLUMNS
    assert rows[0]["unit_id"] == "unit-1"
    assert rows[0]["label"] == "Kitchen"
    assert rows[0]["geom"] == rows[0]["wkt"]
    assert rows[0]["area_m2"] == 6.0


def test_metric_status_does_not_invent_missing_scores(tmp_path) -> None:
    report = tmp_path / "ranked_demo_eval.json"
    report.write_text(json.dumps({
        "checkpoint": "checkpoints/model.pt",
        "mode": "ranked",
        "candidate_budget": 4,
        "n_outlines": 2,
        "n_samples_per_outline": 1,
        "validity": {"outside_frac_mean": 0.0},
    }))
    os.utime(report, None)

    status = load_metric_status(tmp_path)
    assert status.source == str(report)
    assert status.fid is None
    assert status.density is None
    assert status.coverage is None
    assert status.validity == {"outside_frac_mean": 0.0}
    assert status.candidate_budget == 4
    assert status.checkpoint == "checkpoints/model.pt"


def test_metric_status_reads_explicit_scores(tmp_path) -> None:
    report = tmp_path / "post_train_report.json"
    report.write_text(json.dumps({
        "metrics": {
            "fid": 12.5,
            "density": 0.31,
            "coverage": 0.42,
        }
    }))

    status = load_metric_status(tmp_path)
    assert status.fid == 12.5
    assert status.density == 0.31
    assert status.coverage == 0.42
