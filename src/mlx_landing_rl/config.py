"""All knobs in one place, with sane defaults that actually land the rocket."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    """The brain. Small on purpose — 2x128 lands just fine."""

    obs_dim: int = 8
    act_dim: int = 4
    hidden: int = 128


@dataclass
class PPOConfig:
    """PPO hyperparameters. These are the well-worn defaults."""

    gamma: float = 0.99  # how patient: 0.99 cares about the landing, not just now
    lam: float = 0.95  # GAE smoothing: 0.95 = calm, low-noise advantages
    clip_eps: float = 0.2  # PPO speed limit: never move policy >20% at once
    vf_coef: float = 0.5  # critic loss weight
    ent_coef: float = 0.01  # curiosity bonus so engines stay in the mix
    max_grad_norm: float = 0.5  # safety cap on update size
    epochs: int = 4  # passes over each rollout
    batch_size: int = 64  # minibatches from a 2048-step rollout


@dataclass
class TrainConfig:
    """Everything `train-lander` needs. Override from the CLI."""

    total_timesteps: int = 500_000  # full run: ~244 rollouts, 10-15 min on M-series
    rollout_steps: int = 2048  # steps collected before each PPO update
    lr: float = 3e-4  # Adam learning rate
    seed: int = 0
    env_id: str = "LunarLander-v3"
    save_path: str = "checkpoints/lunar_ppo.npz"  # best checkpoint goes here
    model: ModelConfig | None = None
    ppo: PPOConfig | None = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.ppo is None:
            self.ppo = PPOConfig()

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "ppo": asdict(self.ppo),
            **{k: v for k, v in asdict(self).items() if k not in ("model", "ppo")},
        }
