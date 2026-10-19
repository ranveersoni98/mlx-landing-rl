"""Command-line entry points: `train-lander` and `eval-lander`."""

from __future__ import annotations

import argparse


def train_main() -> None:
    """`train-lander --timesteps 500000` — teach the rocket from scratch."""
    from .training import train

    p = argparse.ArgumentParser(
        prog="train-lander",
        description="Train a LunarLander agent with custom PPO in MLX.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="total env steps (500k ≈ 10-15 min on Apple Silicon)",
    )
    p.add_argument(
        "--rollout", type=int, default=2048, help="steps collected before each PPO update"
    )
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument(
        "--save",
        type=str,
        default="checkpoints/lunar_ppo.npz",
        help="where the best checkpoint goes (_final is auto-added)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="start from the weights in --save instead of from scratch",
    )
    p.add_argument("--env", type=str, default="LunarLander-v3", help="gym env id")
    args = p.parse_args()
    train(
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save,
        env_id=args.env,
        resume=args.resume,
    )


def eval_main() -> None:
    """`eval-lander --episodes 10` — watch the trained rocket fly."""
    from .evaluation import evaluate

    p = argparse.ArgumentParser(
        prog="eval-lander",
        description="Evaluate a trained LunarLander checkpoint.",
    )
    p.add_argument("--weights", default="checkpoints/lunar_ppo.npz", help="checkpoint to load")
    p.add_argument("--episodes", type=int, default=5, help="episodes to fly")
    p.add_argument("--render", action="store_true", help="open a window and watch it land")
    p.add_argument(
        "--greedy",
        action="store_true",
        help="use argmax instead of sampling (usually crashes more!)",
    )
    p.add_argument("--env", type=str, default="LunarLander-v3", help="gym env id")
    p.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="PATH",
        help="also save an animated GIF of the flight(s) to PATH (e.g. lander.gif)",
    )
    p.add_argument(
        "--record-episodes",
        type=int,
        default=1,
        metavar="N",
        help="episodes to include in the GIF (default 1 — keep it bite-size)",
    )
    args = p.parse_args()
    evaluate(args.weights, args.episodes, args.render, args.greedy, env_id=args.env)
    if args.record:
        from .evaluation import record

        record(args.weights, args.record, args.record_episodes, args.greedy, env_id=args.env)
