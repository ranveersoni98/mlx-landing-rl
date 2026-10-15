"""The PPO update: how the lander actually gets better.

Short version: for each step, compare the new policy to the old one.
If an action turned out well (positive advantage), nudge its probability up —
but never by more than ~20% at once (that's the "clip"). That speed limit is
the whole trick of PPO: it keeps learning stable instead of veering off.

We also keep the critic honest (predict the returns, with the same clipping)
and add a small entropy bonus so the policy doesn't go deterministic too early.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


def ppo_update(
    model,
    optimizer: optim.Adam,
    obs: np.ndarray,
    acts: np.ndarray,
    old_logp: np.ndarray,
    old_values: np.ndarray,
    advs: np.ndarray,
    rets: np.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    epochs: int = 4,
    batch_size: int = 64,
) -> float:
    """Run a few epochs of clipped PPO over one rollout. Returns mean loss.

    Args:
        model: the ActorCritic to improve.
        optimizer: MLX optimizer (Adam with lr 3e-4 works well here).
        obs, acts, old_logp, old_values: what happened during rollout.
        advs, rets: from `compute_gae` — what we learned from it.
        clip_eps: max policy/value move per update (0.2 = the PPO classic).
        vf_coef: how much the value loss matters vs the policy loss.
        ent_coef: curiosity bonus — keeps all 4 engines in consideration.
        max_grad_norm: safety cap on update size.
        epochs: passes over the rollout (4 is plenty).
        batch_size: minibatch size (64 shuffles well for 2048-step rollouts).
    """
    n = obs.shape[0]

    # Standardizing advantages is a free variance reduction. Always do it.
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            b_obs = mx.array(obs[idx])
            b_act = mx.array(acts[idx].astype(np.int32))
            b_old_logp = mx.array(old_logp[idx])
            b_old_v = mx.array(old_values[idx])
            b_adv = mx.array(advs[idx])
            b_ret = mx.array(rets[idx])

            def loss_fn():
                logits, values = model(b_obs)
                logp_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                new_logp = mx.take_along_axis(logp_all, b_act[:, None], axis=1).squeeze(-1)

                # How much more (or less) likely is this action now vs rollout time?
                ratio = mx.exp(new_logp - b_old_logp)
                pg1 = ratio * b_adv
                pg2 = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                policy_loss = -mx.mean(mx.minimum(pg1, pg2))

                # Critic loss, with the same "don't move too fast" clipping.
                v_clipped = b_old_v + mx.clip(values - b_old_v, -clip_eps, clip_eps)
                v1 = (values - b_ret) ** 2
                v2 = (v_clipped - b_ret) ** 2
                value_loss = 0.5 * mx.mean(mx.maximum(v1, v2))

                # Entropy: reward the policy for keeping its options open.
                probs = mx.softmax(logits, axis=-1)
                entropy = -mx.mean(mx.sum(probs * logp_all, axis=-1))

                return policy_loss + vf_coef * value_loss - ent_coef * entropy

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad()
            grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += float(loss.item())
            num_batches += 1

    return total_loss / max(num_batches, 1)
