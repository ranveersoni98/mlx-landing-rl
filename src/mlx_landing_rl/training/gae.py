"""Generalized Advantage Estimation, in plain NumPy.

The idea: don't just look at the raw reward you got — compare it to what
the critic *expected*. If you did better than expected, that action was
probably good and PPO should do more of it. The `lam` knob blends between
"trust this one step" and "trust the whole future".
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    last_done: bool,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one rollout into advantages + return targets.

    Args:
        rewards: reward at each step, shape (T,).
        values: critic's estimate at each step, shape (T,).
        dones: True where the episode ended (crash / landing / timeout).
        last_value: critic's estimate for the state *after* the rollout
            (used to bootstrap; ignored if the rollout ended in a done).
        last_done: whether that after-rollout state is terminal.
        gamma: how much we discount the future (0.99 = patient).
        lam: GAE smoothing (0.95 = standard, less noisy than raw returns).

    Returns:
        advantages: how much better each step was than expected.
        returns: advantages + values, i.e. what the critic should have said.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)

    # If the rollout ended mid-episode, the future still counts.
    # If it ended exactly on a crash/landing, there's no future to bootstrap.
    next_value = 0.0 if last_done else float(last_value)
    running_gae = 0.0

    for t in reversed(range(n)):
        done = bool(dones[t])
        current_v = float(values[t])

        # Value of the state after t. For the last step that's the
        # bootstrap value; otherwise it's just the next step's estimate.
        next_v = next_value if t == n - 1 else float(values[t + 1])

        # 0 if the episode ended here (no future), else 1.
        non_terminal = 0.0 if done else 1.0
        if t == n - 1:
            next_v = next_value

        # Surprise = "what actually happened" minus "what we expected".
        surprise = rewards[t] + gamma * next_v * non_terminal - current_v
        running_gae = surprise + gamma * lam * non_terminal * running_gae
        advantages[t] = running_gae

        next_value = current_v

    returns = advantages + values.astype(np.float32)
    return advantages, returns
