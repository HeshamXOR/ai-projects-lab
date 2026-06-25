"""Rolling-origin (expanding-window) cross-validation — from scratch.

Repeatedly refits a forecaster on a growing prefix of the series and scores its
multi-step forecast against the held-out window that follows, returning
per-horizon and aggregate error metrics.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from .metrics import mape, mase, smape


class _Forecaster(Protocol):
    """Minimal interface a forecaster must satisfy for backtesting."""

    def fit_forecast(self, series, horizon: int): ...  # noqa: D401, E704


def _to_point(forecast_out) -> np.ndarray:
    """Coerce a forecaster output into a 1-D point-forecast array.

    Accepts either a raw array (HoltWinters/AR) or an object exposing a
    ``point`` attribute (the ensemble's :class:`ForecastResult`).
    """
    if hasattr(forecast_out, "point"):
        return np.asarray(forecast_out.point, dtype=float).ravel()
    return np.asarray(forecast_out, dtype=float).ravel()


def rolling_origin_backtest(
    series: np.ndarray | list[float],
    make_forecaster: Callable[[], _Forecaster],
    horizon: int,
    n_splits: int = 3,
    initial: int | None = None,
    step: int = 1,
    season_length: int = 1,
) -> dict:
    """Expanding-window rolling-origin cross-validation.

    Parameters
    ----------
    series:
        Full observation history.
    make_forecaster:
        Zero-arg factory returning a *fresh* forecaster each origin (so state
        never leaks between folds). Each forecaster must expose
        ``fit_forecast(series, horizon)``.
    horizon:
        Forecast horizon scored at every origin.
    n_splits:
        Number of rolling origins.
    initial:
        Size of the first training window. Defaults to half the series (but at
        least enough to leave ``n_splits`` horizons of test data).
    step:
        How many observations to advance the origin between splits.
    season_length:
        Seasonal period, used for the seasonal-naive scaling inside MASE.

    Returns
    -------
    dict
        ``{"per_horizon": {...}, "aggregate": {...}, "n_origins": int}`` where
        per-horizon holds mean sMAPE at each step ahead and aggregate holds the
        mean MAPE/sMAPE/MASE across all origins and horizons.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = y.size
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    if initial is None:
        initial = max(n // 2, n - n_splits * step - horizon)
    if initial < 2:
        raise ValueError("initial training window too small")
    if initial + horizon > n:
        raise ValueError(
            f"not enough data: initial({initial}) + horizon({horizon}) > n({n})"
        )

    per_horizon_errors: list[np.ndarray] = []
    mape_scores: list[float] = []
    smape_scores: list[float] = []
    mase_scores: list[float] = []
    origins_used = 0

    for s in range(n_splits):
        train_end = initial + s * step
        test_end = train_end + horizon
        if test_end > n:
            break

        train = y[:train_end]
        actual = y[train_end:test_end]

        fc = make_forecaster()
        pred = _to_point(fc.fit_forecast(train, horizon))
        pred = pred[: actual.size]

        # Per-horizon absolute percentage error (symmetric) for this origin.
        denom = np.maximum((np.abs(actual) + np.abs(pred)) / 2.0, 1e-8)
        per_horizon_errors.append(np.abs(actual - pred) / denom)

        mape_scores.append(mape(actual, pred))
        smape_scores.append(smape(actual, pred))
        try:
            mase_scores.append(
                mase(actual, pred, train, season_length=season_length)
            )
        except ValueError:
            # Training window shorter than season_length for early origins.
            pass
        origins_used += 1

    if origins_used == 0:
        raise ValueError("no valid origins produced; check sizes")

    stacked = np.vstack(per_horizon_errors)
    per_horizon = {
        f"h{h + 1}": float(np.mean(stacked[:, h])) for h in range(stacked.shape[1])
    }
    aggregate = {
        "mape": float(np.mean(mape_scores)),
        "smape": float(np.mean(smape_scores)),
        "mase": float(np.mean(mase_scores)) if mase_scores else float("nan"),
    }
    return {
        "per_horizon": per_horizon,
        "aggregate": aggregate,
        "n_origins": origins_used,
    }
