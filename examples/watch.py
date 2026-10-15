"""Pop a window and watch the lander fly. Pure fun, no training.

Run:
    uv run python examples/watch.py
"""

from mlx_landing_rl.evaluation import evaluate

if __name__ == "__main__":
    evaluate(weights="checkpoints/lunar_ppo.npz", episodes=3, render=True)
