"""Seed control utilities for reproducible inverse-problem optimisation.

Provides deterministic seeding for both PyTorch (CPU/GPU) and NumPy, plus
a :func:`seed_everything` convenience and a :func:`jittered_init` helper
that generates a reproducible initial-guess perturbation from a seed.

All random sources in the inverse pipeline (initial guess jitter, observation
noise, mini-batch shuffling if any) must flow through these utilities so that
``seed_everything(s)`` guarantees bit-for-bit reproducibility.
"""

from __future__ import annotations

import math
import os

import torch

__all__ = [
    "jittered_init",
    "seed_everything",
]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all random sources for full reproducibility.

    Args:
        seed: Master seed (any integer).
        deterministic: If True, enable cuDNN deterministic mode and
            disable benchmarking (has a performance cost on GPU).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def jittered_init(
    base_value: float,
    *,
    seed: int,
    relative_range: float = 0.25,
    dtype: torch.dtype = torch.float64,
) -> float:
    """Reproducible multiplicative jitter around ``base_value``.

    Returns ``base_value * (1 + u)`` where ``u ~ Uniform(-range, +range)``.
    The same ``(base_value, seed, relative_range, dtype)`` always returns
    the same number.
    """
    gen = torch.Generator().manual_seed(int(seed) + 104729)
    u = float(2.0 * torch.rand(1, generator=gen, dtype=dtype).item() - 1.0)
    return base_value * (1.0 + relative_range * u)
