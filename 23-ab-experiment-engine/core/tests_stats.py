"""Frequentist hypothesis tests for A/B experiments (from scratch).

WHY THIS MODULE EXISTS
----------------------
An A/B test compares a control arm against one or more treatment arms. Two test
families cover the vast majority of online experiments:

* **Two-proportion z-test** -- for binary / conversion metrics (clicked vs not,
  converted vs not). The estimator is a difference of sample proportions, whose
  sampling distribution is asymptotically normal.
* **Welch's t-test**       -- for continuous metrics (revenue per user, time on
  page) where the two arms may have *unequal variances*. Welch's correction to
  the degrees of freedom is the safe default; the classic Student pooled t-test
  assumes equal variance and is rarely justified in practice.

Both produce a :class:`TestResult` carrying the point estimate, the test
statistic, a two-sided p-value (computed via our from-scratch CDFs in
``core.distributions``), a confidence interval, and a boolean significance flag.

FORMULAS
--------
Two-proportion z (pooled SE for the test statistic)::

    p_hat = (x_a + x_b) / (n_a + n_b)
    SE_pooled = sqrt(p_hat (1 - p_hat) (1/n_a + 1/n_b))
    z = (p_b - p_a) / SE_pooled
    p_value = 2 * (1 - Phi(|z|))

The confidence interval for the *difference* uses the unpooled SE (standard
practice -- the pooled SE is only correct under H0)::

    SE_unpooled = sqrt(p_a(1-p_a)/n_a + p_b(1-p_b)/n_b)
    CI = (p_b - p_a) +/- z_{1-alpha/2} * SE_unpooled

Welch's t::

    se = sqrt(s_a^2/n_a + s_b^2/n_b)
    t  = (mean_b - mean_a) / se
    df = (s_a^2/n_a + s_b^2/n_b)^2
         / ( (s_a^2/n_a)^2/(n_a-1) + (s_b^2/n_b)^2/(n_b-1) )   [Welch-Satterthwaite]
    p_value = 2 * (1 - F_t(|t|; df))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .distributions import normal_cdf, normal_ppf, students_t_cdf, students_t_sf

__all__ = [
    "TestResult",
    "two_proportion_z_test",
    "welch_t_test",
    "lift",
]


@dataclass(frozen=True)
class TestResult:
    """Outcome of a two-arm hypothesis test.

    Attributes
    ----------
    test_name:
        Human-readable name of the test performed.
    estimate:
        Point estimate of the effect (treatment minus control), on the metric's
        natural scale (a proportion difference or a mean difference).
    statistic:
        The z or t test statistic.
    p_value:
        Two-sided p-value.
    df:
        Degrees of freedom (``inf`` for the z-test).
    ci_low, ci_high:
        Lower/upper bounds of the (1 - alpha) confidence interval for the effect.
    alpha:
        Significance level used.
    significant:
        ``True`` when ``p_value < alpha``.
    standard_error:
        Standard error used for the confidence interval.
    relative_lift:
        Effect expressed as a fraction of the control level (``None`` when the
        control level is zero or undefined).
    """

    test_name: str
    estimate: float
    statistic: float
    p_value: float
    df: float
    ci_low: float
    ci_high: float
    alpha: float
    significant: bool
    standard_error: float
    relative_lift: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Serialize to a plain JSON-friendly dictionary."""
        return {
            "test_name": self.test_name,
            "estimate": self.estimate,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "df": (None if math.isinf(self.df) else self.df),
            "ci": [self.ci_low, self.ci_high],
            "alpha": self.alpha,
            "significant": self.significant,
            "standard_error": self.standard_error,
            "relative_lift": self.relative_lift,
            **({"extra": self.extra} if self.extra else {}),
        }


def lift(control_level: float, treatment_level: float) -> float | None:
    """Relative lift = (treatment - control) / control, or ``None`` if control == 0."""
    if control_level == 0.0:
        return None
    return (treatment_level - control_level) / control_level


def two_proportion_z_test(
    conversions_a: int,
    n_a: int,
    conversions_b: int,
    n_b: int,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided two-proportion z-test (arm A = control, arm B = treatment).

    Parameters
    ----------
    conversions_a, n_a:
        Number of conversions and total observations in the control arm.
    conversions_b, n_b:
        Number of conversions and total observations in the treatment arm.
    alpha:
        Significance level (default 0.05 -> 95% CI).

    Returns
    -------
    TestResult
        With ``estimate = p_b - p_a`` (absolute difference in conversion rate).

    Raises
    ------
    ValueError
        If either arm has no observations, or conversion counts are out of range.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("each arm needs at least one observation")
    if not (0 <= conversions_a <= n_a) or not (0 <= conversions_b <= n_b):
        raise ValueError("conversions must be within [0, n] for each arm")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    estimate = p_b - p_a

    # Pooled SE under H0 (used for the test statistic).
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)
    se_pooled = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))

    if se_pooled == 0.0:
        # Degenerate: both arms 0% or both 100%. No detectable effect.
        z = 0.0
        p_value = 1.0
    else:
        z = estimate / se_pooled
        p_value = 2.0 * (1.0 - normal_cdf(abs(z)))

    # Unpooled SE for the confidence interval (correct off the null).
    se_unpooled = math.sqrt(
        p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b
    )
    z_crit = normal_ppf(1.0 - alpha / 2.0)
    margin = z_crit * se_unpooled
    ci_low = estimate - margin
    ci_high = estimate + margin

    return TestResult(
        test_name="two_proportion_z_test",
        estimate=estimate,
        statistic=z,
        p_value=p_value,
        df=math.inf,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant=p_value < alpha,
        standard_error=se_unpooled,
        relative_lift=lift(p_a, p_b),
        extra={
            "rate_control": p_a,
            "rate_treatment": p_b,
            "pooled_rate": p_pool,
            "pooled_standard_error": se_pooled,
            "z_critical": z_crit,
        },
    )


def _mean_var(values: Sequence[float]) -> tuple[float, float, int]:
    """Return (mean, sample variance with n-1 denominator, n)."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 observations to estimate variance")
    mean = math.fsum(values) / n
    ss = math.fsum((v - mean) ** 2 for v in values)
    var = ss / (n - 1)
    return mean, var, n


def welch_t_test(
    sample_a: Sequence[float] | None = None,
    sample_b: Sequence[float] | None = None,
    *,
    mean_a: float | None = None,
    var_a: float | None = None,
    n_a: int | None = None,
    mean_b: float | None = None,
    var_b: float | None = None,
    n_b: int | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's two-sample t-test for unequal variances (A = control, B = treatment).

    You may pass either raw samples (``sample_a``, ``sample_b``) or pre-computed
    summary statistics (``mean_*``, ``var_*``, ``n_*``). Summary statistics let
    the streaming experiment store avoid retaining every observation.

    Returns a :class:`TestResult` with ``estimate = mean_b - mean_a`` and a
    Welch-Satterthwaite degrees-of-freedom estimate.
    """
    if sample_a is not None and sample_b is not None:
        mean_a, var_a, n_a = _mean_var(sample_a)
        mean_b, var_b, n_b = _mean_var(sample_b)
    if None in (mean_a, var_a, n_a, mean_b, var_b, n_b):
        raise ValueError("provide either raw samples or full summary statistics")
    assert n_a is not None and n_b is not None  # for type-checkers
    if n_a < 2 or n_b < 2:
        raise ValueError("each arm needs at least 2 observations")
    if var_a < 0 or var_b < 0:  # type: ignore[operator]
        raise ValueError("variances must be non-negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    se_a_sq = var_a / n_a  # type: ignore[operator]
    se_b_sq = var_b / n_b  # type: ignore[operator]
    se = math.sqrt(se_a_sq + se_b_sq)
    estimate = mean_b - mean_a  # type: ignore[operator]

    if se == 0.0:
        t = 0.0
        df = float(n_a + n_b - 2)
        p_value = 1.0
    else:
        t = estimate / se
        # Welch-Satterthwaite degrees of freedom.
        df = (se_a_sq + se_b_sq) ** 2 / (
            se_a_sq ** 2 / (n_a - 1) + se_b_sq ** 2 / (n_b - 1)
        )
        p_value = 2.0 * students_t_sf(abs(t), df)

    # Confidence interval uses the t critical value at the Welch df.
    t_crit = _t_critical(1.0 - alpha / 2.0, df)
    margin = t_crit * se
    ci_low = estimate - margin
    ci_high = estimate + margin

    return TestResult(
        test_name="welch_t_test",
        estimate=estimate,
        statistic=t,
        p_value=p_value,
        df=df,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant=p_value < alpha,
        standard_error=se,
        relative_lift=lift(mean_a, mean_b),  # type: ignore[arg-type]
        extra={
            "mean_control": mean_a,
            "mean_treatment": mean_b,
            "var_control": var_a,
            "var_treatment": var_b,
            "n_control": n_a,
            "n_treatment": n_b,
            "t_critical": t_crit,
        },
    )


def _t_critical(p: float, df: float, *, tol: float = 1e-10, max_iter: int = 100) -> float:
    """Invert the Student-t CDF: find ``t`` with ``students_t_cdf(t, df) == p``.

    There is no simple closed form, so we bracket-and-bisect using our own t-CDF.
    The normal quantile gives an excellent starting bracket (the t converges to
    the normal as df -> inf), and we widen until the root is bracketed.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    # Symmetric: solve for p >= 0.5 then sign-fold.
    if p < 0.5:
        return -_t_critical(1.0 - p, df, tol=tol, max_iter=max_iter)

    target = p
    lo, hi = 0.0, max(10.0, normal_ppf(p) * 2.0 + 5.0)
    # Expand hi until it brackets the root.
    while students_t_cdf(hi, df) < target and hi < 1e6:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = students_t_cdf(mid, df) - target
        if abs(f) < tol:
            return mid
        if f < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
