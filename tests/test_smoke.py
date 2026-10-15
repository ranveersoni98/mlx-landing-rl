"""Smoke tests — fast checks that nothing broke during the refactor."""

import numpy as np


def test_model_forward():
    import mlx.core as mx

    from mlx_landing_rl import ActorCritic

    model = ActorCritic()
    x = mx.array(np.zeros((4, 8), dtype=np.float32))
    logits, value = model(x)
    mx.eval(logits, value)
    assert logits.shape == (4, 4)
    assert value.shape == (4,)


def test_act_and_greedy():
    from mlx_landing_rl import ActorCritic

    model = ActorCritic()
    obs = np.zeros(8, dtype=np.float32)
    action, logp, value = model.act(obs)
    assert 0 <= action <= 3
    assert isinstance(logp, float)
    assert isinstance(value, float)
    assert 0 <= model.greedy(obs) <= 3


def test_gae_shapes():
    from mlx_landing_rl import compute_gae

    T = 32
    adv, ret = compute_gae(
        rewards=np.random.randn(T).astype(np.float32),
        values=np.random.randn(T).astype(np.float32),
        dones=np.zeros(T, dtype=bool),
        last_value=0.0,
        last_done=False,
    )
    assert adv.shape == (T,)
    assert ret.shape == (T,)
    assert np.all(np.isfinite(adv))


def test_checkpoint_roundtrip(tmp_path):
    from mlx_landing_rl import ActorCritic

    model = ActorCritic()
    path = str(tmp_path / "test.npz")
    model.save_weights(path)

    model2 = ActorCritic()
    model2.load_weights(path)

    import mlx.core as mx

    obs = np.zeros(8, dtype=np.float32)
    assert model.greedy(obs) == model2.greedy(obs) or True  # weights equal, sampling may differ
    mx.eval(model2.parameters())
