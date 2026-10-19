"""Record landings as an animated GIF — for READMEs, PRs, and bragging.

Runs the same loop as `evaluate`, but with `render_mode="rgb_array"` and
captures a frame every `every` steps, then stitches it all into one GIF
with imageio. Sampled actions fly best (same as normal eval), so the
default flight is the one your checkpoint is already good at.
"""

from __future__ import annotations

import os
import platform

import gymnasium as gym
import imageio.v2 as iio
import mlx.core as mx
import numpy as np

from ..models import ActorCritic
from ..utils import load

# Headless boxes (CI, containers) have no display; pygame needs the dummy
# video driver to render RGB frames there. On your Mac nothing changes.
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def record(
    weights: str = "checkpoints/lunar_ppo.npz",
    path: str = "lander.gif",
    episodes: int = 1,
    greedy: bool = False,
    env_id: str = "LunarLander-v3",
    every: int = 2,
    fps: int = 25,
) -> str:
    """Fly `episodes` on camera and save them as one GIF at `path`.

    Args:
        weights: checkpoint to load. Missing file = random policy (with a warning).
        path: output GIF path (e.g. `lander.gif` or `docs/flight.gif`).
        episodes: how many flights to stitch together into one GIF.
        greedy: use argmax instead of sampling (usually crashes more!).
        env_id: gym env id.
        every: keep every Nth frame — the env runs at 50 fps, so `every=2`
            means 25 fps and roughly half the file size.
        fps: GIF playback speed (defaults to 25 to match `every=2`).

    Returns:
        The GIF path, so scripts can print or verify it.
    """
    model = ActorCritic()
    mx.eval(model.parameters())
    load(model, weights)

    env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        step = 0
        done = False
        while not done:
            action = model.greedy(obs) if greedy else model.act(obs)[0]
            obs, _reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            if frame is not None and step % every == 0:
                frames.append(frame)
            step += 1

    env.close()

    # imageio wants duration per frame in ms (deprecates `fps`).
    duration_ms = int(1000 / fps) if fps else 40
    iio.mimsave(path, frames, duration=duration_ms, loop=0)
    print(f"recorded {episodes} episode(s) — {len(frames)} frames -> {path}")
    return path
