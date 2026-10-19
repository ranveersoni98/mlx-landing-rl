# Changelog

## 0.2.0 — 2026-10-19

### Added
- `eval-lander --record lander.gif` — record your pilot's flight as an animated GIF
  (plus `--record-episodes N` to stitch several flights together).
- `train-lander --resume` — warm-start a new run from the best checkpoint in `--save`.
- A real landing captured in `docs/lander.gif` and embedded in the README.

### Changed
- Adam learning rate now anneals **linearly to 0** across the run (standard PPO
  practice — big moves early, gentle settling late). Same starting `lr=3e-4`.

### Fixed
- `seed_all` now also seeds MLX's global RNG, so sampled actions repeat across
  identical runs. (Previously only Python + NumPy were seeded.)
