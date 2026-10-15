"""Watch the trained lander fly — no training here, just vibes and rewards."""

from __future__ import annotations

import gymnasium as gym
import mlx.core as mx
import numpy as np

from ..models import ActorCritic
from ..utils import load


def evaluate(
    weights: str = "checkpoints/lunar_ppo.npz",
    episodes: int = 5,
    render: bool = False,
    greedy: bool = False,
    env_id: str = "LunarLander-v3",
) -> list[float]:
    """Fly `episodes` episodes and report rewards.

    Args:
        weights: checkpoint to load. Missing file = random policy (with a warning).
        episodes: how many landings to attempt.
        render: pop up a window and watch it fly.
        greedy: use argmax instead of sampling. Off by default because the
            released checkpoint lands much better when sampling — greedy
            commits too early and crashes more. Try both and see!
        env_id: gym env id.

    Returns:
        List of total rewards, one per episode. Solved = average >= 200.
    """
    model = ActorCritic()
    mx.eval(model.parameters())
    load(model, weights)

    env = gym.make(env_id, render_mode="human" if render else None)
    rewards: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = model.greedy(obs) if greedy else model.act(obs)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated

        rewards.append(total)
        if total >= 200:
            tag = "LAND"
        elif total < 0:
            tag = "crash"
        else:
            tag = "close"
        print(f"ep {ep + 1}: reward={total:.1f} {tag}")

    mean_rew = float(np.mean(rewards)) if rewards else 0.0
    solved_pct = float(np.mean(np.array(rewards) >= 200) * 100) if rewards else 0.0
    print(f"mean={mean_rew:.1f} solved(>=200)={solved_pct:.0f}%")
    env.close()
    return rewards
