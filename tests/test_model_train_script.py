from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

pytest.importorskip("torch")

from floorgen.model.sampler import load_generator  # noqa: E402
from tests.test_model_data import make_record, write_jsonl  # noqa: E402


def test_train_script_writes_loadable_checkpoint(tmp_path: Path) -> None:
    jsonl = tmp_path / "units.jsonl"
    checkpoint = tmp_path / "flow.pt"
    write_jsonl(jsonl, [make_record(1, "train"), make_record(2, "train")])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_flow.py",
            "--data",
            str(jsonl),
            "--out",
            str(checkpoint),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--boundary-points",
            "8",
            "--hidden",
            "16",
            "--k",
            "4",
            "--device",
            "cpu",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert checkpoint.exists()
    assert '"steps": 1' in result.stdout
    generator = load_generator(checkpoint, steps=1, threshold=-1.0)
    outline = Polygon([(0, 0), (8, 0), (8, 6), (0, 6)])
    mrrs = generator(outline, np.random.default_rng(123))
    assert len(mrrs) <= 4
