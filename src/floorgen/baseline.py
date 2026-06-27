"""Heuristic baseline generator.

IMPORTANT: this is a BASELINE and demo fallback only. The challenge requires the
*scored* generator to be a diffusion/flow-matching model trained from scratch.
This sampler exists to (a) exercise the generate() contract and evaluator before
the model is ready, and (b) give the model a number to beat. It must never be
submitted as the scored generator.

It samples a room count and a label multiset from the empirical MSD
distribution, lays rooms out by recursively slicing the outline's bounding box,
emits axis-aligned MRR tokens, then hands off to the same validity-repair layer
the model uses.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry.base import BaseGeometry

from .config import MAX_ROOMS_K, ROOM_NAME_TO_IDX
from .repr.mrr import RoomMRR

# Empirical priors from the MSD dev subset (label -> rough frequency). Replace
# with values learned in preprocessing for the full run.
LABEL_PRIOR = {
    "Bedroom": 0.24, "Structure": 0.16, "Bathroom": 0.15, "Balcony": 0.12,
    "Corridor": 0.11, "Kitchen": 0.10, "Livingroom": 0.09, "Storeroom": 0.02,
    "Dining": 0.01,
}


def _sample_labels(n: int, rng: np.random.Generator) -> list[int]:
    names = list(LABEL_PRIOR.keys())
    probs = np.array([LABEL_PRIOR[k] for k in names])
    probs = probs / probs.sum()
    # guarantee the essentials, then fill
    picks = ["Kitchen", "Bathroom", "Livingroom"][:n]
    while len(picks) < n:
        picks.append(names[rng.choice(len(names), p=probs)])
    rng.shuffle(picks)
    return [ROOM_NAME_TO_IDX[p] for p in picks]


def _slice_mrrs(outline: BaseGeometry, labels: list[int],
                rng: np.random.Generator) -> list[RoomMRR]:
    """Recursively split the outline bbox into cells and emit MRR tokens.

    Deterministic given the rng. This is baseline scaffolding; the model replaces
    it with learned 5D MRR geometry.
    """
    minx, miny, maxx, maxy = outline.bounds
    cells = [(minx, miny, maxx, maxy)]
    while len(cells) < len(labels):
        # split the largest cell along its longer axis at a jittered midpoint
        cells.sort(key=lambda c: (c[2] - c[0]) * (c[3] - c[1]), reverse=True)
        x0, y0, x1, y1 = cells.pop(0)
        frac = 0.35 + 0.3 * rng.random()
        if (x1 - x0) >= (y1 - y0):
            xm = x0 + (x1 - x0) * frac
            cells += [(x0, y0, xm, y1), (xm, y0, x1, y1)]
        else:
            ym = y0 + (y1 - y0) * frac
            cells += [(x0, y0, x1, ym), (x0, ym, x1, y1)]

    mrrs = []
    for (x0, y0, x1, y1), lab in zip(cells, labels):
        mrrs.append(RoomMRR(
            cx=(x0 + x1) / 2,
            cy=(y0 + y1) / 2,
            w=x1 - x0,
            h=y1 - y0,
            angle=0.0,
            label_idx=lab,
        ))
    return mrrs


def baseline_sample(outline: BaseGeometry, rng: np.random.Generator) -> list[RoomMRR]:
    # room count scales with outline area, clamped to model slot budget
    area = outline.area
    n = int(np.clip(round(3 + area / 14.0), 4, MAX_ROOMS_K))
    labels = _sample_labels(n, rng)
    return _slice_mrrs(outline, labels, rng)
