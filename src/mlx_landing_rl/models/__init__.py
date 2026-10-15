"""Re-export the policy so `from mlx_landing_rl.models import ActorCritic` works."""

from .actor_critic import ActorCritic

__all__ = ["ActorCritic"]
