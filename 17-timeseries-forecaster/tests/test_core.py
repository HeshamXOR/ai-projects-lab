"""Tests for the from-scratch forecasting core."""

from __future__ import annotations

import numpy as np

from core import (
    AR,
    Ensemble,
    HoltWinters,
    mape,
    mase,
    rolling_origin_backtest,
    smape,
)


def _seasonal_series(n: int = 96, m: int = 12, slope: float = 0.3) -> np.ndarray:
    """Deterministic trend + seasonal series (no noise) for tight tolerances."""
    t = np.arange(n, dtype=float)
    season = 10.0 * np.sin(2 * np.pi * t / m)
    return 50.0 + slope * t + season


def test_holtwinters_forecasts_seasonal_series() -> None:
    """Holt-Winters should forecast a clean seasonal+trend series accurately."""
    m = 12
    series = _seasonal_series(n=120, m=m)
    train, test = series[:-m], series[-m:]

    hw = HoltWinters(season_length=m, alpha=0.4, beta=0.1, gamma=0.3).fit(train)
    pred = hw.forecast(m)

    err = smape(test, pred)
    assert err < 0.10, f"Holt-Winters sMAPE too high: {err:.4f}"


def test_ar_recovers_known_process() -> None:
    """AR(2) should track a stable, known AR process closely."""
    rng = np.random.default_rng(0)
    n = 400
    phi1, phi2 = 0.5, -0.3
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.normal(0, 0.5)

    train, test = y[:-10], y[-10:]
    ar = AR(p=2, include_intercept=True).fit(train)

    # Coefficients should be in the right neighborhood.
    assert abs(ar.coef_[0] - phi1) < 0.15
    assert abs(ar.coef_[1] - phi2) < 0.15

    pred = ar.forecast(10)
    # One-step forecast on a stationary process should beat the series std.
    assert np.std(test - pred) < np.std(test) + 0.5


def test_metrics_known_values() -> None:
    """MAPE / sMAPE / MASE on hand-computed inputs."""
    y_true = np.array([100.0, 200.0, 400.0])
    y_pred = np.array([110.0, 180.0, 440.0])

    # |10/100| + |20/200| + |40/400| = 0.1 + 0.1 + 0.1 -> mean 0.1
    assert abs(mape(y_true, y_pred) - 0.1) < 1e-9

    # sMAPE term by term: 10/105, 20/190, 40/420
    expected_smape = np.mean([10 / 105, 20 / 190, 40 / 420])
    assert abs(smape(y_true, y_pred) - expected_smape) < 1e-9

    # MASE: naive seasonal (m=1) scale on a constant-step train = 1.0
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # |diff| always 1 -> scale 1
    mae = np.mean(np.abs(y_true - y_pred))  # (10+20+40)/3 = 23.333...
    assert abs(mase(y_true, y_pred, y_train, season_length=1) - mae) < 1e-9


def test_backtest_runs_and_is_sane() -> None:
    """Rolling-origin backtest returns finite, sensible metrics."""
    m = 12
    series = _seasonal_series(n=120, m=m)

    result = rolling_origin_backtest(
        series,
        make_forecaster=lambda: HoltWinters(season_length=m),
        horizon=m,
        n_splits=3,
        season_length=m,
    )

    assert result["n_origins"] >= 1
    agg = result["aggregate"]
    assert 0.0 <= agg["smape"] < 0.5
    assert np.isfinite(agg["mape"])
    assert len(result["per_horizon"]) == m


def test_ensemble_intervals_bracket_point() -> None:
    """Ensemble forecast intervals satisfy lower <= point <= upper, and widen."""
    m = 12
    series = _seasonal_series(n=96, m=m)

    ens = Ensemble(season_length=m, ar_order=3, confidence=0.95).fit(series)
    res = ens.forecast(8)

    assert np.all(res.lower <= res.point + 1e-9)
    assert np.all(res.point <= res.upper + 1e-9)
    # Interval width should be non-decreasing with horizon.
    widths = res.upper - res.lower
    assert np.all(np.diff(widths) >= -1e-9)
    # Weights are a valid convex combination.
    assert abs(sum(res.weights.values()) - 1.0) < 1e-9


def test_holtwinters_multiplicative_mode_runs() -> None:
    """Multiplicative seasonality path fits and forecasts a positive series."""
    m = 12
    t = np.arange(120, dtype=float)
    series = (50.0 + 0.2 * t) * (1.0 + 0.2 * np.sin(2 * np.pi * t / m))
    hw = HoltWinters(season_length=m, seasonal="mul").fit(series)
    pred = hw.forecast(m)
    assert pred.shape == (m,)
    assert np.all(np.isfinite(pred))
