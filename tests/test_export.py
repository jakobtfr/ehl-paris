"""Smoke tests for floorgen.export — batch export pipeline.

Exercises the export module with simple demo outlines, verifying output schema,
WKT validity, metadata sidecar, and determinism.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from shapely import wkt
from shapely.geometry import box

from floorgen.export import ExportConfig, export_layouts, export_to_csv, export_to_parquet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def demo_outlines() -> dict:
    """Simple rectangular outlines for testing."""
    return {
        "unit_001": box(0, 0, 12, 10),
        "unit_002": box(0, 0, 8, 8),
    }


@pytest.fixture
def export_cfg(tmp_path: Path) -> ExportConfig:
    return ExportConfig(
        output_dir=tmp_path / "export_test",
        n_samples=2,
        seed=42,
        checkpoint="test-baseline",
        config_notes="smoke test run",
    )


# ---------------------------------------------------------------------------
# export_layouts tests
# ---------------------------------------------------------------------------
class TestExportLayouts:
    def test_returns_dataframe(self, demo_outlines, export_cfg):
        df = export_layouts(demo_outlines, export_cfg)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, demo_outlines, export_cfg):
        df = export_layouts(demo_outlines, export_cfg)
        required = {
            "unit_id", "sample_idx", "seed", "label", "label_idx", "geom", "wkt", "area_m2",
        }
        assert required.issubset(set(df.columns))

    def test_wkt_valid(self, demo_outlines, export_cfg):
        df = export_layouts(demo_outlines, export_cfg)
        for wkt_str in df["wkt"].head(5):
            geom = wkt.loads(wkt_str)
            assert geom.is_valid or geom.buffer(0).is_valid
            assert geom.area > 0

    def test_unit_ids_preserved(self, demo_outlines, export_cfg):
        df = export_layouts(demo_outlines, export_cfg)
        exported_ids = set(df["unit_id"].unique())
        assert exported_ids.issubset(set(demo_outlines.keys()))

    def test_processed_metadata_preserved(self, export_cfg):
        outlines = {
            "201": {
                "geometry": box(0, 0, 8, 8),
                "unit_id": 201,
                "plan_id": 2001,
                "floor_id": 9901,
                "split": "test",
                "official_split": "test",
            }
        }
        df = export_layouts(outlines, export_cfg)
        assert set(df["unit_id"]) == {"201"}
        assert set(df["plan_id"]) == {2001}
        assert set(df["floor_id"]) == {9901}
        assert set(df["split"]) == {"test"}
        assert set(df["official_split"]) == {"test"}
        assert df["geom"].equals(df["wkt"])

    def test_sample_indices(self, demo_outlines, export_cfg):
        df = export_layouts(demo_outlines, export_cfg)
        assert df["sample_idx"].max() < export_cfg.n_samples

    def test_validity_columns_present(self, demo_outlines, export_cfg):
        export_cfg_with_validity = ExportConfig(
            output_dir=export_cfg.output_dir,
            n_samples=2,
            seed=42,
            include_validity=True,
        )
        df = export_layouts(demo_outlines, export_cfg_with_validity)
        validity_cols = [c for c in df.columns if c.startswith("v_")]
        assert len(validity_cols) > 0

    def test_deterministic(self, demo_outlines, export_cfg):
        df1 = export_layouts(demo_outlines, export_cfg)
        df2 = export_layouts(demo_outlines, export_cfg)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Parquet export tests
# ---------------------------------------------------------------------------
class TestExportParquet:
    def test_creates_parquet_file(self, demo_outlines, export_cfg):
        path = export_to_parquet(demo_outlines, export_cfg)
        assert path.exists()
        assert path.suffix == ".parquet"

    def test_metadata_sidecar(self, demo_outlines, export_cfg):
        path = export_to_parquet(demo_outlines, export_cfg)
        meta_path = path.with_name(path.stem + "_meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["n_outlines"] == len(demo_outlines)
        assert meta["seed"] == 42
        assert meta["checkpoint"] == "test-baseline"
        assert "columns" in meta

    def test_parquet_readable(self, demo_outlines, export_cfg):
        path = export_to_parquet(demo_outlines, export_cfg)
        df = pd.read_parquet(path)
        assert len(df) > 0


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------
class TestExportCSV:
    def test_creates_csv_file(self, demo_outlines, export_cfg):
        path = export_to_csv(demo_outlines, export_cfg)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_csv_readable(self, demo_outlines, export_cfg):
        path = export_to_csv(demo_outlines, export_cfg)
        df = pd.read_csv(path)
        assert "wkt" in df.columns
        assert len(df) > 0

    def test_export_batch_fails_on_invalid_checkpoint_env(self, tmp_path):
        env = {
            **os.environ,
            "FLOORGEN_CHECKPOINT": str(tmp_path / "missing.pt"),
            "FLOORGEN_PRESENCE_THRESHOLD": "2.0",
        }

        result = subprocess.run(
            [
                sys.executable,
                "scripts/export_batch.py",
                "--demo",
                "--format",
                "csv",
                "--output-dir",
                str(tmp_path / "export"),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "FLOORGEN_PRESENCE_THRESHOLD must be between 0 and 1" in result.stderr
