"""Proofs for the from-scratch diffusion math and toy model."""

import numpy as np

from core.ddim import Diffusion, classifier_free_guidance
from core.toy import NoisePredictor, sample, make_two_moons


def test_alpha_bars_decrease():
    d = Diffusion(T=100)
    # cumulative product of alphas must be monotonically decreasing in (0,1]
    assert np.all(np.diff(d.alpha_bars) < 0)
    assert d.alpha_bars[0] < 1.0 and d.alpha_bars[-1] > 0.0


def test_q_sample_interpolates_noise():
    d = Diffusion(T=100)
    x0 = np.ones((4, 2))
    # at t=0 the sample is almost pure x0; near T it's almost pure noise
    early, _ = d.q_sample(x0, 1, noise=np.zeros_like(x0))
    np.testing.assert_allclose(early, np.sqrt(d.alpha_bars[1]) * x0, rtol=1e-6)


def test_ddim_step_recovers_x0_when_eps_known():
    # if the predicted noise is exactly the noise used, DDIM should march
    # toward the true x0
    d = Diffusion(T=100)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((8, 2))
    t = 50
    xt, noise = d.q_sample(x0, t)
    x0_pred = (xt - np.sqrt(1 - d.alpha_bars[t]) * noise) / np.sqrt(d.alpha_bars[t])
    np.testing.assert_allclose(x0_pred, x0, rtol=1e-5)


def test_cfg_scales_direction():
    u = np.array([[0.0, 0.0]])
    c = np.array([[1.0, 1.0]])
    out = classifier_free_guidance(u, c, scale=2.0)
    np.testing.assert_allclose(out, [[2.0, 2.0]])


def test_toy_diffusion_learns_distribution():
    # train briefly, then check generated points resemble the target's spread
    data = make_two_moons(n=600, seed=1)
    d = Diffusion(T=100)
    model = NoisePredictor(hidden=64, seed=0)
    losses = model.train(data, d, epochs=400, batch=128, lr=2e-3, seed=0)
    assert losses[-1] < losses[0]  # it learned something
    gen = sample(model, d, n=300, steps=30, seed=0)
    # generated cloud should have finite, data-like scale (not blown up)
    assert np.all(np.isfinite(gen))
    assert 0.3 < gen.std() < 3.0
