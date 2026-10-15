"""Load the released lander and fly one episode — the 30-second tour.

Run:
    uv run python examples/quickstart.py
"""

import gymnasium as gym
import mlx.core as mx

from mlx_landing_rl.models import ActorCritic

WEIGHTS = "checkpoints/lunar_ppo.npz"


def main():
    model = ActorCritic()
    mx.eval(model.parameters())
    try:
        model.load_weights(WEIGHTS)
        mx.eval(model.parameters())
        print(f"loaded {WEIGHTS} — flying with the trained pilot")
    except Exception:
        print(f"couldn't find {WEIGHTS}, flying random (it'll crash, that's fine)")

    env = gym.make("LunarLander-v3")
    obs, _ = env.reset()
    total = 0.0
    done = False
    while not done:
        action, _, _ = model.act(obs)  # sampling flies best for this checkpoint
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated

    print(f"episode reward: {total:.1f}  (200+ means nailed the landing)")
    env.close()


if __name__ == "__main__":
    main()
