"""Single source of truth for seeding. `seed_everything(42)` covers python,
numpy, and (if installed) torch. Inference determinism for generate() is the
must-have; full CUDA/ROCm training determinism needs extra backend flags and is
opt-in because it costs speed.
"""

from __future__ import annotations

import os
import random

from .config import SEED


def seed_everything(seed: int = SEED, *, deterministic_torch: bool = False) -> int:
    """Seed all RNGs we depend on. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # ROCm presents as torch.cuda too
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed
