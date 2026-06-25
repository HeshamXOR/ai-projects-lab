"""CUPED variance reduction (Controlled-experiment Using Pre-Experiment Data).

WHY THIS MODULE EXISTS
----------------------
The precision of an A/B test is bounded by the variance of the metric. CUPED
(Deng, Xu, Kohavi & Walker, WSDM 2013 -- the technique Microsoft/Bing popularized)
uses a *pre-experiment covariate* ``X`` -- a quantity measured before treatment
and therefore independent of the assignment -- to soak up variance that has
nothing to do with the treatment. A user who spent a lot last month will likely
spend a lot this month regardless of which arm they land in; subtracting that
predictable component sharpens the estimate of the treatment effect.

THE MATH
--------
Define the adjusted metric::

    Y_cuped = Y - theta * (X - E[X])

where ``theta = Cov(Y, X) / Var(X)`` is the OLS slope of Y on X. Because the
covariate is pre-experiment, ``E[X]`` is identical across arms in expectation, so
the adjustment is *unbiased* for the treatment effect: ``E[Y_cuped] = E[Y]``.

The variance of the adjusted metric is::

    Var(Y_cuped) = Var(Y) * (1 - rho^2)

where ``rho = corr(Y, X)``. So the fractional variance reduction equals
``rho^2`` -- a covariate correlated 0.7 with the metric removes ~49% of the
variance, equivalent to ~doubling the effective sample size for free.

This module computes ``theta`` from the pooled data (both arms share one slope,
the standard recipe), produces adjusted per-arm means/variances, and reports the
achieved variance reduction so a caller can feed the adjusted summaries straight
into :func:`core.tests_stats.welch_t_test`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["CupedResult", "compute_theta", "apply_cuped"]


@dataclass(frozen=True)
class CupedResult:
    """Adjusted arm summaries produced by CUPED.

    Attributes
    ----------
    theta:
        OLS slope Cov(Y, X) / Var(X) used for the adjustment.
    grand_mean_x:
        Pooled mean of the covariate (the centering constant).
    adjusted_mean_a / adjusted_mean_b:
        Means of the adjusted metric in each arm (unbiased for raw means).
    adjusted_var_a / adjusted_var_b:
        Sample variances of the adjusted metric in each arm.
    raw_var_a / raw_var_b:
        Sample variances of the raw metric in each arm (for comparison).
    n_a / n_b:
        Arm sizes.
    variance_reduction:
        Pooled fractional variance reduction = rho^2 in [0, 1].
    correlation:
        Pooled Pearson correlation between Y and X.
    """

    theta: float
    grand_mean_x: float
    adjusted_mean_a: float
    adjusted_mean_b: float
    adjusted_var_a: float
    adjusted_var_b: float
    raw_var_a: float
    raw_var_b: float
    n_a: int
    n_b: int
    variance_reduction: float
    correlation: float

    def as_dict(self) -> dict:
        return {
            "theta": self.theta,
            "grand_mean_x": self.grand_mean_x,
            "adjusted_mean_control": self.adjusted_mean_a,
            "adjusted_mean_treatment": self.adjusted_mean_b,
            "adjusted_var_control": self.adjusted_var_a,
            "adjusted_var_treatment": self.adjusted_var_b,
            "raw_var_control": self.raw_var_a,
            "raw_var_treatment": self.raw_var_b,
            "n_control": self.n_a,
            "n_treatment": self.n_b,
            "variance_reduction": self.variance_reduction,
            "correlation": self.correlation,
        }


def compute_theta(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Return (theta, grand_mean_x, pooled_correlation) from pooled Y and X.

    theta = Cov(Y, X) / Var(X). Covariance/variance use the population (ddof=0)
    convention here because theta is a ratio and the ``n`` cancels; the
    correlation is reported alongside so callers can see how strong the covariate
    relationship is.
    """
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape")
    if y.size < 2:
        raise ValueError("need at least 2 paired observations")

    mean_y = float(np.mean(y))
    mean_x = float(np.mean(x))
    dy = y - mean_y
    dx = x - mean_x

    var_x = float(np.mean(dx * dx))
    if var_x == 0.0:
        raise ValueError("covariate has zero variance; CUPED is undefined")

    cov_xy = float(np.mean(dy * dx))
    theta = cov_xy / var_x

    var_y = float(np.mean(dy * dy))
    corr = 0.0 if var_y == 0.0 else cov_xy / math.sqrt(var_x * var_y)

    return theta, mean_x, corr


def apply_cuped(
    y_a: Sequence[float],
    x_a: Sequence[float],
    y_b: Sequence[float],
    x_b: Sequence[float],
) -> CupedResult:
    """Apply CUPED to two arms given their metric (Y) and pre-experiment covariate (X).

    Parameters
    ----------
    y_a, x_a:
        Control-arm metric values and paired covariate values.
    y_b, x_b:
        Treatment-arm metric values and paired covariate values.

    Returns
    -------
    CupedResult
        Adjusted per-arm means and variances plus the achieved variance
        reduction. The adjusted summaries can be passed directly to
        :func:`core.tests_stats.welch_t_test`.

    Notes
    -----
    A single ``theta`` is estimated from the *pooled* data (both arms), which is
    the standard CUPED recipe and avoids using arm-specific slopes that could
    leak the treatment effect into the adjustment.
    """
    ya = np.asarray(y_a, dtype=float)
    xa = np.asarray(x_a, dtype=float)
    yb = np.asarray(y_b, dtype=float)
    xb = np.asarray(x_b, dtype=float)

    if ya.shape != xa.shape or yb.shape != xb.shape:
        raise ValueError("each arm's Y and X must be paired (same length)")
    if ya.size < 2 or yb.size < 2:
        raise ValueError("each arm needs at least 2 observations")

    y_all = np.concatenate([ya, yb])
    x_all = np.concatenate([xa, xb])
    theta, grand_mean_x, corr = compute_theta(y_all, x_all)

    # Adjusted metric per arm: Y - theta * (X - grand_mean_x).
    adj_a = ya - theta * (xa - grand_mean_x)
    adj_b = yb - theta * (xb - grand_mean_x)

    # Sample variances (ddof=1) for downstream t-tests.
    raw_var_a = float(np.var(ya, ddof=1))
    raw_var_b = float(np.var(yb, ddof=1))
    adj_var_a = float(np.var(adj_a, ddof=1))
    adj_var_b = float(np.var(adj_b, ddof=1))

    variance_reduction = corr * corr  # rho^2 (theoretical pooled reduction)

    return CupedResult(
        theta=theta,
        grand_mean_x=grand_mean_x,
        adjusted_mean_a=float(np.mean(adj_a)),
        adjusted_mean_b=float(np.mean(adj_b)),
        adjusted_var_a=adj_var_a,
        adjusted_var_b=adj_var_b,
        raw_var_a=raw_var_a,
        raw_var_b=raw_var_b,
        n_a=int(ya.size),
        n_b=int(yb.size),
        variance_reduction=variance_reduction,
        correlation=corr,
    )
