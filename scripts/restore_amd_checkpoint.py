"""Restore the AMD checkpoint from Git-tracked binary chunks.

GitHub LFS upload is unavailable for this repository, so the trained AMD
checkpoint is stored as fixed-size chunks under checkpoints/*.pt.parts/.
This script reconstructs the original .pt file and verifies its checksum.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTS_DIR = ROOT / "checkpoints" / "flow-transformer-amd-862d422.pt.parts"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "flow-transformer-amd-862d422.pt"
EXPECTED_SHA256 = "d47d6e083e65e301c44fd1ecb40b8c1326316fff73b4b39712bfe54d53fb70ca"
EXPECTED_SIZE = 369_429_793


def iter_parts(parts_dir: Path) -> list[Path]:
    parts = sorted(parts_dir.glob("part-*"))
    if not parts:
        raise FileNotFoundError(f"No checkpoint chunks found in {parts_dir}")
    return parts


def restore(parts_dir: Path, output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total_size = 0

    with output.open("wb") as dest:
        for part in iter_parts(parts_dir):
            with part.open("rb") as src:
                while chunk := src.read(1024 * 1024):
                    dest.write(chunk)
                    digest.update(chunk)
                    total_size += len(chunk)

    actual_sha256 = digest.hexdigest()
    if total_size != EXPECTED_SIZE or actual_sha256 != EXPECTED_SHA256:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "Restored checkpoint failed verification: "
            f"size={total_size}, sha256={actual_sha256}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    restore(args.parts_dir, args.output, args.force)
    print(f"Restored {args.output}")
    print(f"sha256={EXPECTED_SHA256}")
    print(f"size={EXPECTED_SIZE}")


if __name__ == "__main__":
    main()
