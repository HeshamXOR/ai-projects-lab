"""Statistical power and sample-size calculations (from scratch).

WHY THIS MODULE EXISTS
----------------------
Before launching an experiment you must answer: "How many users per arm do I
need to reliably detect an effect of size X?" Running underpowered tests is the
single most common A/B-testing mistake -- you spend traffic and conclude "no
effect" when the experiment never had a chance of detecting one.

The classic two-sided formula for comparing two proportions (control rate ``p``
vs treatment rate ``p + mde``) is::

    n_per_arm = ( z_{1-alpha/2} * sqrt(2 * p_bar (1 - p_bar))
                  + z_{1-beta} * sqrt(p(1-p) + q(1-q)) )^2  /  (q - p)^2

where ``q = p + mde``, ``p_bar = (p + q) / 2``, and ``z_{1-beta}`` is the
quantile for the desired power. Everything hinges on the inverse normal CDF
(:func:`core.distributions.normal_ppf`), which we implemented ourselves.

We also provide the inverse direction -- :func:`power_for_sample_size` -- which
reports the achieved power given a fixed ``n``, and a continuous-metric variant
:func:`sample_size_for_mean` for t-test-style experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .distributions import normal_cdf, normal_ppf

__all__ = [
    "PowerResult",
    "sample_size_two_proportion",
    "power_for_sample_size",
    "sample_size_for_mean",
]


@dataclass(frozen=True)
class PowerResult:
    """Result of a power / sample-size computation."""

    required_n_per_arm: int
    baseline_rate: float
    mde_absolute: float
    treatment_rate: float
    alpha: float
    power: float
    z_alpha: float
    z_beta: float

    def as_dict(self) -> dict:
        return {
            "required_n_per_arm": self.required_n_per_arm,
            "required_n_total": self.required_n_per_arm * 2,
            "baseline_rate": self.baseline_rate,
            "mde_absolute": self.mde_absolute,
            "treatment_rate": self.treatment_rate,
            "alpha": self.alpha,
            "power": self.power,
            "z_alpha": self.z_alpha,
            "z_beta": self.z_beta,
        }


def sample_size_two_proportion(
    baseline_rate: float,
    mde_absolute: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> PowerResult:
    """Required sample size *per arm* to detect an absolute lift ``mde_absolute``.

    Parameters
    ----------
    baseline_rate:
        Control conversion rate ``p`` in (0, 1).
    mde_absolute:
        Minimum detectable effect, expressed as an *absolute* change in rate.
        E.g. detecting a move from 0.10 to 0.12 -> ``mde_absolute = 0.02``.
    alpha:
        Type-I error rate (false positive). Default 0.05.
    power:
        Desired power = 1 - beta (true positive). Default 0.80.
    two_sided:
        If ``True`` use ``z_{1-alpha/2}``; otherwise ``z_{1-alpha}``.

    Returns
    -------
    PowerResult

    Raises
    ------
    ValueError
        For out-of-range rates or a treatment rate that leaves (0, 1).
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_absolute == 0.0:
        raise ValueError("mde_absolute must be non-zero")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")

    p = baseline_rate
    q = baseline_rate + mde_absolute
    if not 0.0 < q < 1.0:
        raise ValueError("treatment rate (baseline + mde) must stay within (0, 1)")

    p_bar = (p + q) / 2.0
    z_alpha = normal_ppf(1.0 - alpha / 2.0) if two_sided else normal_ppf(1.0 - alpha)
    z_beta = normal_ppf(power)

    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * math.sqrt(p * (1.0 - p) + q * (1.0 - q))
    )
    n = (numerator ** 2) / ((q - p) ** 2)
    n_per_arm = int(math.ceil(n))

    return PowerResult(
        required_n_per_arm=n_per_arm,
        baseline_rate=p,
        mde_absolute=mde_absolute,
        treatment_rate=q,
        alpha=alpha,
        power=power,
        z_alpha=z_alpha,
        z_beta=z_beta,
    )


def power_for_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Achieved power for a *given* per-arm sample size.

    Inverts the sample-size relation: solve for ``z_beta`` and map through the
    normal CDF. Useful to annotate a running experiment with "you currently have
    62% power to detect the configured MDE".
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")

    p = baseline_rate
    q = baseline_rate + mde_absolute
    if not 0.0 < q < 1.0:
        raise ValueError("treatment rate (baseline + mde) must stay within (0, 1)")

    p_bar = (p + q) / 2.0
    z_alpha = normal_ppf(1.0 - alpha / 2.0) if two_sided else normal_ppf(1.0 - alpha)

    sd_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    sd_alt = math.sqrt(p * (1.0 - p) + q * (1.0 - q))

    z_beta = (abs(q - p) * math.sqrt(n_per_arm) - z_alpha * sd_null) / sd_alt
    return normal_cdf(z_beta)


def sample_size_for_mean(
    sigma: float,
    mde_absolute: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Required per-arm sample size for a continuous metric (equal-variance approx).

    n_per_arm = 2 * sigma^2 * (z_{1-alpha/2} + z_{1-beta})^2 / mde^2

    This is the normal-approximation sample size used to plan t-test
    experiments; for large ``n`` the t-distribution converges to the normal so
    the approximation is excellent.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if mde_absolute == 0.0:
        raise ValueError("mde_absolute must be non-zero")

    z_alpha = normal_ppf(1.0 - alpha / 2.0) if two_sided else normal_ppf(1.0 - alpha)
    z_beta = normal_ppf(power)
    n = 2.0 * sigma ** 2 * (z_alpha + z_beta) ** 2 / (mde_absolute ** 2)
    return int(math.ceil(n))
