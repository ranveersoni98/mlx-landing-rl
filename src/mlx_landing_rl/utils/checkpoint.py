"""Checkpoint helpers — saving and loading the lander's brain."""

from __future__ import annotations

import mlx.core as mx


def save(model, path: str) -> str:
    """Save weights to `path`. Creates parent folders if needed."""
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    model.save_weights(path)
    return path


def load(model, path: str, verbose: bool = True) -> bool:
    """Load weights into `model`. Returns True on success, False if missing.

    We return a bool (instead of raising) so eval scripts can fall back
    to a random policy with a friendly message when no checkpoint exists yet.
    """
    try:
        model.load_weights(path)
        mx.eval(model.parameters())
        if verbose:
            print(f"loaded {path}")
        return True
    except Exception as e:
        if verbose:
            print(f"no weights at {path} ({e}), using random policy")
        return False
