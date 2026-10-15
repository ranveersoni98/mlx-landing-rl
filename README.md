# mlx-landing-rl

**Teach a little rocket to land itself**

> Released weights: **[RanveerSoni98/mlx-landing-rl](https://huggingface.co/RanveerSoni98/mlx-landing-rl)**

![Python](https://img.shields.io/badge/python-3.12-blue)
![MLX](https://img.shields.io/badge/mlx-apple_silicon-black)
![PPO from scratch](https://img.shields.io/badge/PPO-from_scratch-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What is this?

You know LunarLander that old arcade game where you have to land a wobbly rocket with limited fuel? This repo trains an AI pilot to do exactly that.

- **World:** `LunarLander-v3` from Gymnasium (with Box2D physics)
- **Brain:** a tiny actor-critic net (`8 → 128 → 128 → 4 + value`) in [MLX](https://github.com/ml-explore/mlx)
- **Learning:** custom PPO — no Stable-Baselines, no shortcuts, just the algorithm written out so you can actually read it
- **Result:** mean reward ~200+ ("solved!") after ~500k steps

The agent sees 8 numbers (position, velocity, angle, leg contact) and picks one of 4 things each step: do nothing, fire left, fire main, fire right. Every landing earns +100–140 points, every crash costs −100, and burning the main engine costs a little fuel each step. Simple rules, surprisingly hard to master.

## Try it in 2 minutes

```bash
# 1. Grab it
git clone https://github.com/RanveerSoni98/mlx-landing-rl.git
cd mlx-landing-rl
uv sync

# 2. Watch the pre-trained pilot land (no training needed)
uv run eval-lander --episodes 5

# 3. Actually watch it — opens a window, very satisfying
uv run eval-lander --episodes 3 --render

# 4. Or run the tiny demo script
uv run python examples/quickstart.py
```

Expected output looks like this:

```
loaded checkpoints/lunar_ppo.npz
ep 1: reward=253.1 LAND
ep 2: reward=247.6 LAND
...
mean=215.4 solved(>=200)=80%
```

## Train your own

```bash
# Quick sanity check (~30 seconds, won't solve anything — just proves it runs)
uv run train-lander --timesteps 4096 --rollout 1024 --save checkpoints/smoke.npz

# The real thing (~10-15 min on M1/M2/M3/M4, ~500k steps)
uv run train-lander --timesteps 500000

# Then evaluate your new pilot
uv run eval-lander --weights checkpoints/lunar_ppo.npz --episodes 10
```

## How it actually learns

If you've never touched RL, here's the whole idea in 4 lines:

1. The policy flies 2048 steps, recording every state, action, and reward.
2. We figure out which actions were *better than expected* (that's the "advantage", computed with GAE).
3. We nudge the policy toward good actions — but at most ~20% at a time (that's the PPO "clip", the one trick that keeps training stable).
4. Repeat ~244 times. Keep the best checkpoint.

Files that do each part, in plain English:

| Step | File | What it does |
|------|------|--------------|
| Brain | `src/mlx_landing_rl/models/actor_critic.py` | The neural net — shared trunk, actor head + critic head |
| Advantages | `src/mlx_landing_rl/training/gae.py` | GAE: "was that action better than expected?" |
| Learning | `src/mlx_landing_rl/training/ppo.py` | Clipped PPO update + value loss + entropy bonus |
| Loop | `src/mlx_landing_rl/training/trainer.py` | Fly, learn, save best, repeat |
| Watching | `src/mlx_landing_rl/evaluation/evaluate.py` | Fly episodes, print LAND or crash |
| Knobs | `src/mlx_landing_rl/config.py` | Every hyperparameter in one dataclass |

Default hyperparameters (the boring ones that just work): `γ=0.99, λ=0.95, clip=0.2, lr=3e-4, 4 epochs, batch 64, entropy 0.01, value loss 0.5`.

## Project layout

Everything lives in folders because flat files are chaos:

```
mlx-landing-rl/
├── src/mlx_landing_rl/          # the actual package
│   ├── __init__.py              # what `import mlx_landing_rl` gives you
│   ├── config.py                # all knobs in one place (dataclasses)
│   ├── cli.py                   # train-lander / eval-lander entry points
│   ├── models/
│   │   └── actor_critic.py      # the brain
│   ├── training/
│   │   ├── gae.py               # advantages
│   │   ├── ppo.py               # the PPO update
│   │   └── trainer.py           # the fly-learn-repeat loop
│   ├── evaluation/
│   │   └── evaluate.py          # watch it fly
│   └── utils/
│       ├── checkpoint.py        # save / load weights
│       └── seeding.py           # reproducibility
├── checkpoints/
│   ├── lunar_ppo.npz            # best pilot (release weights)
│   └── lunar_ppo_final.npz      # last pilot (for comparison)
├── examples/
│   ├── quickstart.py            # fly one episode in 10 lines
│   └── watch.py                 # open a window, enjoy
├── tests/
│   └── test_smoke.py            # fast sanity checks
├── pyproject.toml
└── README.md
```

## Use it from Python

```python
from mlx_landing_rl.models import ActorCritic
import mlx.core as mx

model = ActorCritic()
model.load_weights("checkpoints/lunar_ppo.npz")
mx.eval(model.parameters())

action, logp, value = model.act(obs)  # obs = 8-number state from gym
```

Or straight from Hugging Face:

```python
from huggingface_hub import hf_hub_download
from mlx_landing_rl.models import ActorCritic
import mlx.core as mx

path = hf_hub_download("RanveerSoni98/mlx-landing-rl", "lunar_ppo.npz")
model = ActorCritic()
model.load_weights(path)
mx.eval(model.parameters())
```

## Results

Trained for 500k steps on Apple Silicon, seed 0:

| Checkpoint | Mean (5 eps, sampling) | Solved rate | Notes |
|------------|----------------------|-------------|-------|
| `lunar_ppo.npz` (best) | ~215 | ~60–80% | Best by training reward — **this is the release** |
| `lunar_ppo_final.npz` | ~190 | ~60% | Last update — slightly worse, kept for comparison |

"Solved" for LunarLander officially means averaging ≥200 over 100 episodes. This pilot gets there — some flights still crash (it's a rocket, cut it some slack), but most stick the landing.

Greedy (argmax) mode only solves ~20% — sampling is the way to fly this one.

## Citation

```bibtex
@software{soni2026mlxlandingrl,
  author = {Ranveer, Soni},
  title = {mlx-landing-rl: LunarLander PPO from scratch in Apple MLX},
  year = {2026},
  url = {https://github.com/RanveerSoni98/mlx-landing-rl}
}
```

## License

Code is MIT - see [LICENSE](LICENSE).

<p align="center">
  Built with <3 and a lot of bad decisions! by <a href="https://ranveersoni.me">Ranveer Soni</a>
</p>
