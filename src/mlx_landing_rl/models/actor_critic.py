"""The brain of the lander.

A small shared-torso actor-critic: one trunk reads the 8-number state,
then two heads disagree politely — the actor picks engines, the critic
guesses how well we're doing.

Nothing fancy here on purpose. Two layers of 128 is plenty to land
a little rocket, and it trains in minutes on Apple Silicon.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """Decides which engine to fire and how good the situation looks.

    State (8 numbers from LunarLander-v3):
        x, y, vx, vy, angle, angular velocity, left-leg touching, right-leg touching

    Actions (4 choices):
        0 = do nothing, 1 = left engine, 2 = main engine, 3 = right engine
    """

    def __init__(self, obs_dim: int = 8, act_dim: int = 4, hidden: int = 128):
        super().__init__()
        # Keep these names stable — old checkpoints load by layer name.
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Run a batch of states through the net.

        Returns:
            logits: unnormalized scores for each of the 4 actions.
            value: predicted future reward from this state.
        """
        h = nn.relu(self.fc1(x))
        h = nn.relu(self.fc2(h))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Pick one action by sampling the policy.

        Sampling (not argmax) is how this checkpoint flies best — the policy
        stays a little exploratory, which helps it recover from wobbles.

        Returns:
            action: 0-3 engine choice.
            logp: log-probability of that action (needed for PPO).
            value: critic's estimate (needed for advantage calculation).
        """
        x = mx.array(np.asarray(obs, dtype=np.float32)[None, :])
        logits, value = self(x)
        mx.eval(logits, value)

        act_arr = mx.random.categorical(logits, axis=-1)
        mx.eval(act_arr)
        action = int(act_arr.item())

        logp_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logp_all)
        logp = float(logp_all[0, action].item())
        v = float(value[0].item())
        return action, logp, v

    def greedy(self, obs: np.ndarray) -> int:
        """Pick the most likely action (argmax). Useful for debugging.

        Note: for the released checkpoint, sampling via `act()` lands far
        more reliably than greedy. Greedy tends to commit too early.
        """
        x = mx.array(np.asarray(obs, dtype=np.float32)[None, :])
        logits, _ = self(x)
        mx.eval(logits)
        return int(mx.argmax(logits, axis=-1).item())
