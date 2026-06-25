"""FastAPI service for multi-horizon time-series forecasting.

Wraps the from-scratch NumPy forecasters (Holt-Winters, AR, ensemble) behind a
small JSON API with Pydantic validation, structured error handling and a
backtest-derived metrics summary.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from core import AR, Ensemble, HoltWinters, rolling_origin_backtest

app = FastAPI(
    title="timeseries-forecaster",
    version="1.0.0",
    description="Multi-horizon forecasting (Holt-Winters / AR / ensemble), "
    "pure-NumPy core.",
)

ModelName = Literal["holtwinters", "ar", "ensemble"]


class ForecastRequest(BaseModel):
    """Request body for ``POST /forecast``."""

    series: list[float] = Field(
        ..., description="Historical observations, oldest first."
    )
    horizon: int = Field(..., gt=0, description="Number of future steps to forecast.")
    season_length: int = Field(
        1, ge=1, description="Seasonal period (1 disables seasonality)."
    )
    model: ModelName = Field("ensemble", description="Which forecaster to use.")
    confidence: float = Field(
        0.95, gt=0.0, lt=1.0, description="Prediction-interval confidence level."
    )
    ar_order: int = Field(2, ge=1, description="AR order p (AR / ensemble).")

    @field_validator("series")
    @classmethod
    def _finite_series(cls, v: list[float]) -> list[float]:
        arr = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("series must contain only finite numbers")
        return v

    @model_validator(mode="after")
    def _check_lengths(self) -> "ForecastRequest":
        n = len(self.series)
        if n < 2:
            raise ValueError("series must have at least 2 observations")
        if self.season_length > 1 and n < 2 * self.season_length:
            raise ValueError(
                f"series length ({n}) must be >= 2 * season_length "
                f"({2 * self.season_length}) for seasonal models"
            )
        if self.model in ("ar", "ensemble") and n <= self.ar_order:
            raise ValueError(
                f"series length ({n}) must be > ar_order ({self.ar_order})"
            )
        return self


class ForecastResponse(BaseModel):
    """Response body for ``POST /forecast``."""

    point: list[float]
    lower: list[float]
    upper: list[float]
    model: str
    metrics: dict | None = None
    weights: dict[str, float] | None = None


def _intervals_from_residuals(
    point: np.ndarray, residuals: np.ndarray, confidence: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build horizon-widening prediction intervals from residual std."""
    from core.ensemble import _z_for  # local import to reuse the z-table

    sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    z = _z_for(confidence)
    steps = np.arange(1, horizon + 1, dtype=float)
    half = z * sigma * np.sqrt(steps)
    return point - half, point + half


@app.get("/")
def root() -> dict:
    """Service metadata and available endpoints."""
    return {
        "service": "timeseries-forecaster",
        "version": "1.0.0",
        "models": ["holtwinters", "ar", "ensemble"],
        "endpoints": {
            "POST /forecast": "produce a multi-horizon forecast",
            "GET /health": "liveness probe",
        },
        "core": "pure-NumPy (Holt-Winters, AR, ensemble) implemented from scratch",
    }


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    """Produce a point forecast with prediction intervals and a metrics summary.

    Raises
    ------
    HTTPException
        ``400`` if the underlying forecaster rejects the inputs at fit time.
    """
    series = np.asarray(req.series, dtype=float)
    h = req.horizon

    try:
        if req.model == "holtwinters":
            hw = HoltWinters(season_length=req.season_length).fit(series)
            point = hw.forecast(h)
            lower, upper = _intervals_from_residuals(
                point, hw.residuals_, req.confidence, h
            )
            weights = None
        elif req.model == "ar":
            ar = AR(p=req.ar_order).fit(series)
            point = ar.forecast(h)
            lower, upper = _intervals_from_residuals(
                point, ar.residuals_, req.confidence, h
            )
            weights = None
        else:  # ensemble
            ens = Ensemble(
                season_length=req.season_length,
                ar_order=req.ar_order,
                confidence=req.confidence,
            ).fit(series)
            result = ens.forecast(h)
            point, lower, upper = result.point, result.lower, result.upper
            weights = result.weights
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Backtest metrics are best-effort: skip if the series is too short.
    metrics: dict | None = None
    try:
        metrics = rolling_origin_backtest(
            series,
            make_forecaster=lambda: _make(req),
            horizon=min(h, max(1, len(series) // 4)),
            n_splits=2,
            season_length=req.season_length,
        )["aggregate"]
    except ValueError:
        metrics = None

    return ForecastResponse(
        point=[float(x) for x in point],
        lower=[float(x) for x in lower],
        upper=[float(x) for x in upper],
        model=req.model,
        metrics=metrics,
        weights=weights,
    )


def _make(req: ForecastRequest):
    """Factory used by the backtester to build a fresh forecaster per origin."""
    if req.model == "holtwinters":
        return HoltWinters(season_length=req.season_length)
    if req.model == "ar":
        return AR(p=req.ar_order)
    return Ensemble(
        season_length=req.season_length,
        ar_order=req.ar_order,
        confidence=req.confidence,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
