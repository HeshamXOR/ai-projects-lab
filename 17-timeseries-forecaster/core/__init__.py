"""From-scratch time-series forecasting core (pure NumPy).

Exposes the forecasters, metrics and backtester so callers can do
``from core import HoltWinters, AR, Ensemble``.
"""

from __future__ import annotations

from .ar import AR
from .backtest import rolling_origin_backtest
from .ensemble import Ensemble, ForecastResult
from .holtwinters import HoltWinters
from .metrics import mape, mase, smape

__all__ = [
    "AR",
    "HoltWinters",
    "Ensemble",
    "ForecastResult",
    "rolling_origin_backtest",
    "mape",
    "smape",
    "mase",
]
