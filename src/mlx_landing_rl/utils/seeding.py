"""Seed everything so runs are repeatable."""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np


def seed_all(seed: int = 0) -> None:
    """Seed Python, NumPy and MLX's global RNG (the sampler in `act()` reads it)."""
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
