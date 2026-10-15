"""Seed everything so runs are repeatable."""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int = 0) -> None:
    """Seed Python, NumPy (MLX sampling follows its own key schedule)."""
    random.seed(seed)
    np.random.seed(seed)
