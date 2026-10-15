"""Training subpackage: rollout collection + GAE + PPO updates."""

from .gae import compute_gae
from .ppo import ppo_update
from .trainer import train

__all__ = ["compute_gae", "ppo_update", "train"]
