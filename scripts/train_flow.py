"""Train or smoke-test the conditional flow model.

Example:
    uv run --extra train python scripts/train_flow.py \
        --data data/processed/units.jsonl --out checkpoints/flow-smoke.pt \
        --epochs 1 --batch-size 2 --max-steps 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from floorgen.config import MAX_ROOMS_K, ROOM_NAMES, SEED
from floorgen.model import (
    FloorRecordDataset,
    RoomFlowModel,
    collate_model_batch,
    conditional_flow_matching_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the floorgen conditional flow model.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/units.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/flow.pt"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no explicit cap.")
    parser.add_argument("--limit", type=int, default=0, help="0 means use all records.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=MAX_ROOMS_K)
    parser.add_argument("--boundary-points", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda device, but torch.cuda.is_available() is false")
    return torch.device(requested)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.boundary_points <= 0:
        raise ValueError("--boundary-points must be positive")
    if args.hidden <= 0:
        raise ValueError("--hidden must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(SEED)
    device = resolve_device(args.device)
    dataset = FloorRecordDataset(
        args.data,
        split=args.split,
        k=args.k,
        boundary_points=args.boundary_points,
        limit=args.limit or None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"no records found for split={args.split!r} in {args.data}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_model_batch,
    )
    model = RoomFlowModel(
        num_types=len(ROOM_NAMES),
        k=args.k,
        d_model=args.hidden,
        boundary_points=args.boundary_points,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    started = time.time()
    initial_loss: float | None = None
    final_loss: float | None = None
    steps = 0
    for _epoch in range(args.epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = conditional_flow_matching_loss(model, batch)
            if not torch.isfinite(result.total):
                raise RuntimeError(f"non-finite loss: {result.detached_scalars()}")
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_value = float(result.total.detach().cpu())
            initial_loss = loss_value if initial_loss is None else initial_loss
            final_loss = loss_value
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
            if args.max_steps and steps >= args.max_steps:
                break

    if steps == 0:
        raise RuntimeError("training produced zero optimizer steps; no checkpoint was saved")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "num_types": len(ROOM_NAMES),
            "k": args.k,
            "d_model": args.hidden,
            "boundary_points": args.boundary_points,
        },
        "label_names": ROOM_NAMES,
        "train": {
            "data": str(args.data),
            "split": args.split,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "steps": steps,
            "lr": args.lr,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "wall_seconds": time.time() - started,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    torch.save(checkpoint, args.out)
    print(json.dumps({"checkpoint": str(args.out), **checkpoint["train"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
