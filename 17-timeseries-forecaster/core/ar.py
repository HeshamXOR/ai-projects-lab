"""Autoregressive AR(p) model — implemented from scratch in NumPy.

Fits an AR(p) model by ordinary least squares: build a design matrix of lagged
observations, then solve the normal equations via ``np.linalg.lstsq``. Forecasts
are produced recursively, feeding predictions back in as new lags.
"""

from __future__ import annotations

import numpy as np


class AR:
    """Autoregressive model of order ``p`` fitted by least squares.

    The model is::

        y_t = c + phi_1 * y_{t-1} + ... + phi_p * y_{t-p} + e_t

    where ``c`` is an intercept (optional) and ``phi`` are the AR coefficients.

    Parameters
    ----------
    p:
        Autoregressive order (number of lags), ``>= 1``.
    include_intercept:
        Whether to fit a constant term.
    """

    def __init__(self, p: int = 1, include_intercept: bool = True) -> None:
        if p < 1:
            raise ValueError("p must be >= 1")
        self.p = int(p)
        self.include_intercept = bool(include_intercept)

        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None  # shape (p,), lag-1 first
        self.history_: np.ndarray | None = None
        self.fitted_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.sigma_: float = 0.0

    # -- design matrix -----------------------------------------------------

    def _design(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build the lagged design matrix ``X`` and target ``Y``.

        Row ``i`` of ``X`` holds ``[y_{i+p-1}, ..., y_i]`` (lag-1 .. lag-p) and
        ``Y[i] = y_{i+p}``. An intercept column of ones is prepended when
        ``include_intercept`` is set.
        """
        n = y.size
        rows = n - self.p
        X = np.empty((rows, self.p), dtype=float)
        for lag in range(1, self.p + 1):
            # Column for lag ``lag`` aligns y_{t-lag} with target y_t.
            X[:, lag - 1] = y[self.p - lag : n - lag]
        Y = y[self.p :]
        if self.include_intercept:
            X = np.hstack([np.ones((rows, 1), dtype=float), X])
        return X, Y

    # -- fitting -----------------------------------------------------------

    def fit(self, series: np.ndarray | list[float]) -> "AR":
        """Fit AR(p) coefficients by least squares.

        Parameters
        ----------
        series:
            1-D array of observations, length ``> p``.

        Returns
        -------
        self
        """
        y = np.asarray(series, dtype=float).ravel()
        if y.size <= self.p:
            raise ValueError(f"series length must be > p={self.p}, got {y.size}")

        X, Y = self._design(y)
        # Solve the normal equations via least squares (stable, handles rank
        # deficiency through the SVD inside lstsq).
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)

        if self.include_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:].astype(float)
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.astype(float)

        self.fitted_ = X @ beta
        self.residuals_ = Y - self.fitted_
        self.sigma_ = float(np.std(self.residuals_, ddof=1)) if Y.size > 1 else 0.0
        self.history_ = y[-self.p :].copy()
        return self

    # -- forecasting -------------------------------------------------------

    def forecast(self, horizon: int) -> np.ndarray:
        """Recursive multi-step forecast.

        Each step uses the latest ``p`` values — observed history at first, then
        previously forecast values — to predict the next point.

        Parameters
        ----------
        horizon:
            Number of future steps (``>= 1``).

        Returns
        -------
        np.ndarray
            Forecast array of length ``horizon``.
        """
        if self.coef_ is None or self.history_ is None:
            raise RuntimeError("call fit() before forecast()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        # ``buf`` holds the most recent p values, lag-1 at index -1.
        buf = list(self.history_)
        out = np.empty(horizon, dtype=float)
        for h in range(horizon):
            # lags ordered lag-1 .. lag-p
            lags = np.array(buf[-1 : -self.p - 1 : -1], dtype=float)
            pred = self.intercept_ + float(self.coef_ @ lags)
            out[h] = pred
            buf.append(pred)
        return out

    def fit_forecast(
        self, series: np.ndarray | list[float], horizon: int
    ) -> np.ndarray:
        """Convenience: ``fit(series)`` then ``forecast(horizon)``."""
        return self.fit(series).forecast(horizon)
