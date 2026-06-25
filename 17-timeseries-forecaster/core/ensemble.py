"""Ensemble forecaster — combines Holt-Winters and AR with prediction intervals.

Fits both base models, blends their point forecasts (equal or inverse-error
weighting based on in-sample residual error), and derives prediction intervals
from the combined residual standard deviation that widen with the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .ar import AR
from .holtwinters import HoltWinters

WeightMode = Literal["equal", "inverse_error"]


@dataclass
class ForecastResult:
    """Container for an ensemble forecast.

    Attributes
    ----------
    point:
        Point forecast, length ``horizon``.
    lower, upper:
        Lower/upper prediction-interval bounds (same length).
    model:
        Identifier of the producing model.
    weights:
        Mapping of base-model name to its blend weight.
    sigma:
        Combined residual standard deviation used for the intervals.
    """

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model: str
    weights: dict[str, float]
    sigma: float


# Normal-approximation z-scores for common confidence levels.
_Z_TABLE = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def _z_for(confidence: float) -> float:
    """Return the z-score for a confidence level (nearest tabulated value)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    key = min(_Z_TABLE, key=lambda k: abs(k - confidence))
    return _Z_TABLE[key]


class Ensemble:
    """Blend of :class:`HoltWinters` and :class:`AR` with prediction intervals.

    Parameters
    ----------
    season_length:
        Seasonal period passed to Holt-Winters.
    ar_order:
        Order ``p`` of the AR component.
    weighting:
        ``"equal"`` averages the two forecasts; ``"inverse_error"`` weights each
        model by the inverse of its in-sample residual MSE (better in-sample fit
        gets more weight).
    confidence:
        Confidence level for the prediction interval (e.g. ``0.95``).
    seasonal:
        Seasonal mode for Holt-Winters (``"add"`` or ``"mul"``).
    """

    def __init__(
        self,
        season_length: int = 1,
        ar_order: int = 2,
        weighting: WeightMode = "inverse_error",
        confidence: float = 0.95,
        seasonal: str = "add",
    ) -> None:
        self.season_length = int(season_length)
        self.ar_order = int(ar_order)
        self.weighting = weighting
        self.confidence = float(confidence)
        self.seasonal = seasonal

        self.hw_: HoltWinters | None = None
        self.ar_: AR | None = None
        self.weights_: dict[str, float] = {}
        self.sigma_: float = 0.0

    def _safe_ar_order(self, n: int) -> int:
        """Cap AR order so the design matrix has enough rows to fit."""
        # Need at least a couple of rows beyond the parameters.
        return max(1, min(self.ar_order, n - 2))

    def fit(self, series: np.ndarray | list[float]) -> "Ensemble":
        """Fit both base models and compute blend weights + residual sigma."""
        y = np.asarray(series, dtype=float).ravel()

        hw = HoltWinters(
            season_length=self.season_length, seasonal=self.seasonal
        ).fit(y)
        ar = AR(p=self._safe_ar_order(y.size)).fit(y)
        self.hw_, self.ar_ = hw, ar

        hw_mse = float(np.mean(hw.residuals_**2)) if hw.residuals_ is not None else 1.0
        ar_mse = float(np.mean(ar.residuals_**2)) if ar.residuals_ is not None else 1.0
        eps = 1e-12

        if self.weighting == "equal":
            w_hw = w_ar = 0.5
        else:
            inv_hw = 1.0 / max(hw_mse, eps)
            inv_ar = 1.0 / max(ar_mse, eps)
            total = inv_hw + inv_ar
            w_hw, w_ar = inv_hw / total, inv_ar / total

        self.weights_ = {"holtwinters": w_hw, "ar": w_ar}
        # Combined residual std: weighted MSE -> sigma.
        combined_var = w_hw * hw_mse + w_ar * ar_mse
        self.sigma_ = float(np.sqrt(max(combined_var, 0.0)))
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        """Blend base forecasts and build horizon-widening intervals.

        The interval half-width at step ``h`` is ``z * sigma * sqrt(h)`` — the
        usual random-walk-style widening of uncertainty with the horizon.
        """
        if self.hw_ is None or self.ar_ is None:
            raise RuntimeError("call fit() before forecast()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        f_hw = self.hw_.forecast(horizon)
        f_ar = self.ar_.forecast(horizon)
        w_hw = self.weights_["holtwinters"]
        w_ar = self.weights_["ar"]
        point = w_hw * f_hw + w_ar * f_ar

        z = _z_for(self.confidence)
        steps = np.arange(1, horizon + 1, dtype=float)
        half_width = z * self.sigma_ * np.sqrt(steps)
        lower = point - half_width
        upper = point + half_width

        return ForecastResult(
            point=point,
            lower=lower,
            upper=upper,
            model="ensemble",
            weights=dict(self.weights_),
            sigma=self.sigma_,
        )

    def fit_forecast(
        self, series: np.ndarray | list[float], horizon: int
    ) -> ForecastResult:
        """Convenience: ``fit(series)`` then ``forecast(horizon)``."""
        return self.fit(series).forecast(horizon)
