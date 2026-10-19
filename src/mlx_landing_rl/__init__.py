"""mlx-landing-rl — teach a rocket to land itself with PPO in Apple MLX.

No demos, no imitation — the agent learns purely from trial, error,
and the occasional fiery crash.
"""

from .config import ModelConfig, PPOConfig, TrainConfig
from .evaluation import evaluate, record
from .models import ActorCritic
from .training import compute_gae, ppo_update, train

__version__ = "0.2.0"

__all__ = [
    "ActorCritic",
    "ModelConfig",
    "PPOConfig",
    "TrainConfig",
    "compute_gae",
    "evaluate",
    "ppo_update",
    "record",
    "train",
]
