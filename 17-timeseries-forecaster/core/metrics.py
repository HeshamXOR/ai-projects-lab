"""Forecast metrics — MAPE, sMAPE, MASE implemented from scratch in NumPy."""

from __future__ import annotations

import numpy as np


def _as_arrays(
    y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty input")
    return yt, yp


def mape(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error (as a fraction, multiply by 100 for %).

    ``MAPE = mean(|y_true - y_pred| / |y_true|)``. ``eps`` guards against
    division by zero for near-zero actuals.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yt - yp) / denom))


def smape(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    eps: float = 1e-8,
) -> float:
    """Symmetric MAPE (as a fraction in ``[0, 1]``).

    ``sMAPE = mean( |y_true - y_pred| / ((|y_true| + |y_pred|) / 2) )``.

    Symmetric in the sense that over- and under-forecasts of the same magnitude
    are penalized equally; bounded, unlike plain MAPE.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
    return float(np.mean(np.abs(yt - yp) / denom))


def mase(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    y_train: np.ndarray | list[float],
    season_length: int = 1,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Scaled Error.

    The forecast MAE is scaled by the in-sample MAE of a **seasonal naive**
    forecast on the training series::

        scale = mean(|y_train_t - y_train_{t-m}|)   for t = m .. n
        MASE  = mean(|y_true - y_pred|) / scale

    ``MASE < 1`` means the forecast beats seasonal naive; ``> 1`` means worse.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    tr = np.asarray(y_train, dtype=float).ravel()
    m = int(season_length)
    if m < 1:
        raise ValueError("season_length must be >= 1")
    if tr.size <= m:
        raise ValueError(f"y_train length must be > season_length={m}")

    naive_errors = np.abs(tr[m:] - tr[:-m])
    scale = float(np.mean(naive_errors))
    scale = max(scale, eps)
    return float(np.mean(np.abs(yt - yp)) / scale)
