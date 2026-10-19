"""The training loop: fly, learn, repeat until it lands.

Each iteration:
  1. Fly 2048 steps with the current policy, recording everything.
  2. Turn rewards into advantages with GAE (see `gae.py`).
  3. Run PPO updates (see `ppo.py`).
  4. Keep the best checkpoint by episode reward.

A full 500k-step run is ~244 updates and takes 10-15 minutes on Apple Silicon.
"""

from __future__ import annotations

import math
import os
import time

import gymnasium as gym
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from ..config import TrainConfig
from ..models import ActorCritic
from ..utils import seed_all
from .gae import compute_gae
from .ppo import ppo_update


def train(
    total_timesteps: int = 500_000,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    seed: int = 0,
    save_path: str = "checkpoints/lunar_ppo.npz",
    log_every: int = 1,
    env_id: str = "LunarLander-v3",
    resume: bool = False,
) -> str:
    """Train the lander. With `resume=True`, warm-starts from `save_path`."""
    cfg = TrainConfig(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        lr=lr,
        seed=seed,
        env_id=env_id,
        save_path=save_path,
    )
    return train_from_config(cfg, log_every=log_every, resume=resume)


def train_from_config(cfg: TrainConfig, log_every: int = 1, resume: bool = False) -> str:
    """Same as `train()`, but takes a `TrainConfig` — handy for sweeps."""
    seed_all(cfg.seed)

    env = gym.make(cfg.env_id)
    model = ActorCritic(
        obs_dim=cfg.model.obs_dim,
        act_dim=cfg.model.act_dim,
        hidden=cfg.model.hidden,
    )
    mx.eval(model.parameters())

    if resume:
        # `load` prints a friendly note and returns False if there's no file,
        # so resuming with an empty folder just quietly starts from scratch.
        from ..utils import load

        if load(model, cfg.save_path):
            print(f"resuming from {cfg.save_path}")

    # Standard PPO practice: anneal the learning rate to 0 across the run.
    # Early updates make big adjustments; late ones are gentle pruning so the
    # policy settles instead of wobbling around the optimum.
    num_updates = cfg.total_timesteps // cfg.rollout_steps
    batches_per_update = cfg.ppo.epochs * math.ceil(cfg.rollout_steps / cfg.ppo.batch_size)
    total_opt_steps = max(num_updates * batches_per_update, 1)
    lr_schedule = optim.linear_schedule(cfg.lr, 0.0, steps=total_opt_steps)
    optimizer = optim.Adam(learning_rate=lr_schedule)

    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

    obs, _ = env.reset(seed=cfg.seed)
    # Tracks whether the state carried into the next rollout is terminal,
    # so GAE knows whether bootstrapping is valid.
    next_done = False
    global_step = 0
    num_updates = cfg.total_timesteps // cfg.rollout_steps
    best_mean = -1e9
    t0 = time.time()

    obs_dim = cfg.model.obs_dim
    for update in range(1, num_updates + 1):
        # One rollout's worth of memories.
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros(cfg.rollout_steps, dtype=np.int32)
        logp_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.rollout_steps, dtype=np.bool_)

        ep_rews: list[float] = []
        ep_rew = 0.0

        # --- Fly ---------------------------------------------------------
        for s in range(cfg.rollout_steps):
            obs_buf[s] = obs
            action, logp, value = model.act(obs)
            act_buf[s] = action
            logp_buf[s] = logp
            val_buf[s] = value

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            rew_buf[s] = reward
            done_buf[s] = done
            ep_rew += reward
            global_step += 1

            if done:
                ep_rews.append(ep_rew)
                obs, _ = env.reset()
                ep_rew = 0.0
                next_done = True
            else:
                next_done = False

        # --- Learn -------------------------------------------------------
        # Value of the state we'd continue from — GAE's "what comes next".
        _, last_v = model(mx.array(np.asarray(obs, dtype=np.float32)[None, :]))
        mx.eval(last_v)
        last_value = float(last_v[0].item())

        adv, ret = compute_gae(
            rew_buf,
            val_buf,
            done_buf,
            last_value,
            next_done,
            gamma=cfg.ppo.gamma,
            lam=cfg.ppo.lam,
        )
        loss = ppo_update(
            model,
            optimizer,
            obs_buf,
            act_buf,
            logp_buf,
            val_buf,
            adv,
            ret,
            clip_eps=cfg.ppo.clip_eps,
            vf_coef=cfg.ppo.vf_coef,
            ent_coef=cfg.ppo.ent_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            batch_size=cfg.ppo.batch_size,
        )

        # --- Keep the best ----------------------------------------------
        mean_rew = float(np.mean(ep_rews)) if ep_rews else float("nan")
        if mean_rew == mean_rew and mean_rew > best_mean:  # not nan
            best_mean = mean_rew
            model.save_weights(cfg.save_path)

        if update % log_every == 0:
            solved = "SOLVED!" if mean_rew >= 200 else ""
            print(
                f"update {update}/{num_updates} steps={global_step} "
                f"loss={loss:.4f} ep_rew={mean_rew:.1f} "
                f"n_ep={len(ep_rews)} best={best_mean:.1f} "
                f"{solved} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    final_path = cfg.save_path.replace(".npz", "_final.npz")
    model.save_weights(final_path)
    print(f"saved best -> {cfg.save_path}  |  last -> {final_path}")
    env.close()
    return cfg.save_path
