"""Holt-Winters exponential smoothing — implemented from scratch in NumPy.

Triple exponential smoothing (level + trend + seasonality) with the classic
additive recursions, plus an optional multiplicative seasonal variant. No
statsmodels — the level/trend/seasonal recursions, seasonal initialization and
multi-step forecast are all hand-rolled here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SeasonalMode = Literal["add", "mul"]


class HoltWinters:
    """Holt-Winters (triple) exponential smoothing forecaster.

    Parameters
    ----------
    season_length:
        Number of observations per seasonal cycle (``m``). Use ``1`` to disable
        seasonality (falls back to double exponential smoothing — Holt's linear
        trend).
    alpha, beta, gamma:
        Smoothing parameters for level, trend and seasonal components, each in
        ``[0, 1]``.
    trend:
        Whether to include a trend component. If ``False`` the model is simple
        single/seasonal smoothing.
    seasonal:
        ``"add"`` for additive seasonality or ``"mul"`` for multiplicative.

    Notes
    -----
    Additive recursions (with trend ``b`` and seasonal ``s`` of period ``m``)::

        level_t   = alpha * (y_t - s_{t-m}) + (1 - alpha) * (level_{t-1} + b_{t-1})
        b_t       = beta  * (level_t - level_{t-1}) + (1 - beta) * b_{t-1}
        s_t       = gamma * (y_t - level_t) + (1 - gamma) * s_{t-m}

    The ``h``-step-ahead forecast is ``level_T + h * b_T + s_{T-m+1+((h-1) mod m)}``.
    """

    def __init__(
        self,
        season_length: int = 1,
        alpha: float = 0.5,
        beta: float = 0.1,
        gamma: float = 0.1,
        trend: bool = True,
        seasonal: SeasonalMode = "add",
    ) -> None:
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        for name, val in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if seasonal not in ("add", "mul"):
            raise ValueError("seasonal must be 'add' or 'mul'")

        self.season_length = int(season_length)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.trend = bool(trend)
        self.seasonal = seasonal

        # Fitted state (populated by ``fit``).
        self.level_: float | None = None
        self.trend_: float = 0.0
        self.season_: np.ndarray | None = None
        self.fitted_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None

    # -- initialization ----------------------------------------------------

    def _init_components(self, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Initialize level, trend and seasonal indices from the first cycles.

        Uses the standard approach: the initial level is the mean of the first
        full season; the initial trend is the average per-step difference
        between the first two seasons; seasonal indices come from the deviation
        (additive) or ratio (multiplicative) of the first season to its mean.
        """
        m = self.season_length
        if m == 1:
            level0 = float(y[0])
            trend0 = float(y[1] - y[0]) if (self.trend and y.size >= 2) else 0.0
            return level0, trend0, np.array([0.0] if self.seasonal == "add" else [1.0])

        first = y[:m]
        level0 = float(np.mean(first))

        if self.trend and y.size >= 2 * m:
            second = y[m : 2 * m]
            trend0 = float((np.mean(second) - np.mean(first)) / m)
        else:
            trend0 = 0.0

        if self.seasonal == "add":
            season0 = first - level0
        else:
            # Guard against division by zero in multiplicative mode.
            level0_safe = level0 if abs(level0) > 1e-12 else 1e-12
            season0 = first / level0_safe
        return level0, trend0, season0.astype(float)

    # -- fitting -----------------------------------------------------------

    def fit(self, series: np.ndarray | list[float]) -> "HoltWinters":
        """Run the smoothing recursions over ``series`` and store final state.

        Parameters
        ----------
        series:
            1-D array of observations, length ``>= 2 * season_length`` when
            seasonality is enabled.

        Returns
        -------
        self
        """
        y = np.asarray(series, dtype=float).ravel()
        m = self.season_length
        if y.size < max(2, 2 * m if m > 1 else 2):
            raise ValueError(
                f"series too short: need >= {max(2, 2 * m)} points for "
                f"season_length={m}, got {y.size}"
            )

        level, trend, season = self._init_components(y)
        season = season.copy()
        fitted = np.empty(y.size, dtype=float)

        for t in range(y.size):
            s_idx = t % m
            seasonal_comp = season[s_idx]

            # One-step-ahead in-sample forecast (before observing y[t]).
            if self.seasonal == "add":
                fitted[t] = level + (trend if self.trend else 0.0) + seasonal_comp
            else:
                fitted[t] = (level + (trend if self.trend else 0.0)) * seasonal_comp

            prev_level = level
            if self.seasonal == "add":
                level = self.alpha * (y[t] - seasonal_comp) + (1 - self.alpha) * (
                    prev_level + (trend if self.trend else 0.0)
                )
            else:
                comp = seasonal_comp if abs(seasonal_comp) > 1e-12 else 1e-12
                level = self.alpha * (y[t] / comp) + (1 - self.alpha) * (
                    prev_level + (trend if self.trend else 0.0)
                )

            if self.trend:
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend

            if m > 1:
                if self.seasonal == "add":
                    season[s_idx] = self.gamma * (y[t] - level) + (
                        1 - self.gamma
                    ) * seasonal_comp
                else:
                    lvl = level if abs(level) > 1e-12 else 1e-12
                    season[s_idx] = self.gamma * (y[t] / lvl) + (
                        1 - self.gamma
                    ) * seasonal_comp

        self.level_ = float(level)
        self.trend_ = float(trend) if self.trend else 0.0
        self.season_ = season
        self.fitted_ = fitted
        self.residuals_ = y - fitted
        return self

    # -- forecasting -------------------------------------------------------

    def forecast(self, horizon: int) -> np.ndarray:
        """Produce an ``horizon``-step-ahead point forecast.

        Parameters
        ----------
        horizon:
            Number of future steps (``>= 1``).

        Returns
        -------
        np.ndarray
            Array of length ``horizon`` with the multi-step forecast.
        """
        if self.level_ is None or self.season_ is None:
            raise RuntimeError("call fit() before forecast()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        m = self.season_length
        out = np.empty(horizon, dtype=float)
        for h in range(1, horizon + 1):
            base = self.level_ + (h * self.trend_ if self.trend else 0.0)
            if m > 1:
                s = self.season_[(h - 1) % m]
            else:
                s = 0.0 if self.seasonal == "add" else 1.0
            out[h - 1] = base + s if self.seasonal == "add" else base * s
        return out

    def fit_forecast(
        self, series: np.ndarray | list[float], horizon: int
    ) -> np.ndarray:
        """Convenience: ``fit(series)`` then ``forecast(horizon)``."""
        return self.fit(series).forecast(horizon)
