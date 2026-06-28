from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from floorgen.model.network import RoomFlowModel  # noqa: E402
from floorgen.posttrain import (  # noqa: E402
    checkpoint_sha256,
    load_outlines_from_units,
    run_post_training,
)
from tests.test_model_data import make_record, write_jsonl  # noqa: E402


def write_checkpoint(path: Path) -> None:
    model = RoomFlowModel(k=4, d_model=16, boundary_points=8)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "num_types": 10,
                "k": 4,
                "d_model": 16,
                "boundary_points": 8,
            },
            "label_names": ["Bedroom"],
            "train": {"steps": 1, "final_loss": 1.23},
        },
        path,
    )


def test_load_outlines_from_units_filters_split_and_limit(tmp_path: Path) -> None:
    units = tmp_path / "units.jsonl"
    write_jsonl(
        units,
        [
            make_record(1, "train"),
            make_record(2, "val"),
            make_record(3, "val"),
        ],
    )

    outlines = load_outlines_from_units(units, split="val", limit=1)

    assert list(outlines) == ["2"]
    assert outlines["2"].area > 0


def test_run_post_training_writes_report_and_export(tmp_path: Path, monkeypatch) -> None:
    units = tmp_path / "units.jsonl"
    checkpoint = tmp_path / "flow.pt"
    output_dir = tmp_path / "post_train"
    write_jsonl(units, [make_record(1, "val")])
    write_checkpoint(checkpoint)

    import floorgen.generate as generate_module
    import floorgen.posttrain as posttrain_module
    from floorgen.baseline import baseline_sample

    monkeypatch.setattr(
        posttrain_module,
        "load_generator",
        lambda *_args, **_kwargs: baseline_sample,
    )
    old_generator = generate_module.GENERATOR
    try:
        result = run_post_training(
            checkpoint_path=checkpoint,
            units_path=units,
            output_dir=output_dir,
            split="val",
            n_samples=1,
            steps=1,
            threshold=0.0,
            export_format="csv",
        )
    finally:
        generate_module.GENERATOR = old_generator

    assert result.checkpoint_sha256 == checkpoint_sha256(checkpoint)
    assert result.score is not None
    assert result.score["n_outlines"] == 1
    assert result.export_path is not None and Path(result.export_path).exists()
    assert result.report_path is not None and Path(result.report_path).exists()
    assert result.markdown_path is not None and Path(result.markdown_path).exists()
    report = json.loads(Path(result.report_path).read_text())
    assert report["checkpoint_train"]["steps"] == 1
    assert result.score["room_count_mean"] > 0


def test_run_post_training_rejects_invalid_threshold(tmp_path: Path) -> None:
    units = tmp_path / "units.jsonl"
    checkpoint = tmp_path / "flow.pt"
    write_jsonl(units, [make_record(1, "val")])
    write_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        run_post_training(
            checkpoint_path=checkpoint,
            units_path=units,
            split="val",
            steps=1,
            threshold=2.0,
            dry_run=True,
        )


def test_run_post_training_rejects_empty_generated_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    units = tmp_path / "units.jsonl"
    checkpoint = tmp_path / "flow.pt"
    write_jsonl(units, [make_record(1, "val")])
    write_checkpoint(checkpoint)

    import floorgen.generate as generate_module
    import floorgen.posttrain as posttrain_module

    old_generator = generate_module.GENERATOR
    try:
        monkeypatch.setattr(
            posttrain_module,
            "load_generator",
            lambda *_args, **_kwargs: (lambda _outline, _rng: []),
        )
        with pytest.raises(RuntimeError, match="generated no rooms"):
            run_post_training(
                checkpoint_path=checkpoint,
                units_path=units,
                output_dir=tmp_path / "post_train",
                split="val",
                n_samples=1,
                steps=1,
                threshold=1.0,
                export_format="csv",
            )
    finally:
        generate_module.GENERATOR = old_generator


def test_post_train_cli_dry_run_json(tmp_path: Path) -> None:
    units = tmp_path / "units.jsonl"
    checkpoint = tmp_path / "flow.pt"
    write_jsonl(units, [make_record(1, "val")])
    write_checkpoint(checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/post_train.py",
            "--checkpoint",
            str(checkpoint),
            "--units",
            str(units),
            "--split",
            "val",
            "--steps",
            "1",
            "--threshold",
            "0.0",
            "--dry-run",
            "--json",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["n_outlines"] == 1
    assert payload["checkpoint_train"]["steps"] == 1


def test_generate_env_loader_rejects_invalid_threshold(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow.pt"
    write_checkpoint(checkpoint)
    env = {
        **os.environ,
        "FLOORGEN_CHECKPOINT": str(checkpoint),
        "FLOORGEN_SAMPLE_STEPS": "1",
        "FLOORGEN_PRESENCE_THRESHOLD": "2.0",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from shapely.geometry import box; "
                "from floorgen.generate import generate; "
                "generate(box(0, 0, 8, 6))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "FLOORGEN_PRESENCE_THRESHOLD must be between 0 and 1" in result.stderr


def test_post_train_cli_explicit_checkpoint_ignores_stale_env_checkpoint(tmp_path: Path) -> None:
    units = tmp_path / "units.jsonl"
    checkpoint = tmp_path / "flow.pt"
    write_jsonl(units, [make_record(1, "val")])
    write_checkpoint(checkpoint)
    env = {
        **os.environ,
        "FLOORGEN_CHECKPOINT": str(tmp_path / "missing.pt"),
        "FLOORGEN_PRESENCE_THRESHOLD": "2.0",
    }

    result = subprocess.run(
        [
            sys.executable,
            "scripts/post_train.py",
            "--checkpoint",
            str(checkpoint),
            "--units",
            str(units),
            "--split",
            "val",
            "--steps",
            "1",
            "--threshold",
            "0.0",
            "--dry-run",
            "--json",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["checkpoint"] == str(checkpoint)
