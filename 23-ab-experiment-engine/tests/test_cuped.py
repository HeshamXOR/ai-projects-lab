"""Tests for CUPED variance reduction (core.cuped) on constructed data."""

from __future__ import annotations

import numpy as np

from core.cuped import apply_cuped, compute_theta


def _make_correlated(n: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct Y, X with population correlation ~rho.

    X ~ N(0,1); Y = rho*X + sqrt(1-rho^2)*eps, eps ~ N(0,1) independent.
    Then corr(Y, X) ~= rho and Var(Y) ~= 1.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    y = rho * x + np.sqrt(1.0 - rho * rho) * eps
    return y, x


def test_theta_recovers_slope():
    # If Y = 3 X + noise, theta = Cov(Y,X)/Var(X) should be ~3.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    y = 3.0 * x + 0.5 * rng.standard_normal(20000)
    theta, mean_x, corr = compute_theta(y, x)
    assert abs(theta - 3.0) < 0.05
    assert abs(mean_x) < 0.05
    assert corr > 0.95


def test_cuped_reduces_variance():
    # Strongly correlated covariate (rho ~0.8) should cut variance substantially.
    y_a, x_a = _make_correlated(8000, rho=0.8, seed=1)
    y_b, x_b = _make_correlated(8000, rho=0.8, seed=2)
    # Add a real treatment effect to arm B so the adjustment must stay unbiased.
    y_b = y_b + 0.10

    res = apply_cuped(y_a, x_a, y_b, x_b)

    # Adjusted variance must be strictly below raw variance in BOTH arms.
    assert res.adjusted_var_a < res.raw_var_a
    assert res.adjusted_var_b < res.raw_var_b

    # Theoretical reduction is rho^2 ~= 0.64; allow sampling slack.
    assert 0.55 < res.variance_reduction < 0.72

    # CUPED is unbiased: adjusted means stay close to raw means.
    assert abs(res.adjusted_mean_a - float(np.mean(y_a))) < 0.05
    assert abs(res.adjusted_mean_b - float(np.mean(y_b))) < 0.05


def test_cuped_preserves_treatment_effect():
    # Construct arms differing by a known effect of 0.25; CUPED should still
    # recover roughly that difference in adjusted means.
    y_a, x_a = _make_correlated(10000, rho=0.7, seed=10)
    y_b, x_b = _make_correlated(10000, rho=0.7, seed=11)
    y_b = y_b + 0.25
    res = apply_cuped(y_a, x_a, y_b, x_b)
    effect = res.adjusted_mean_b - res.adjusted_mean_a
    assert abs(effect - 0.25) < 0.05


def test_cuped_weak_covariate_small_reduction():
    # A nearly-uncorrelated covariate yields little variance reduction.
    y_a, x_a = _make_correlated(8000, rho=0.05, seed=20)
    y_b, x_b = _make_correlated(8000, rho=0.05, seed=21)
    res = apply_cuped(y_a, x_a, y_b, x_b)
    assert res.variance_reduction < 0.05
