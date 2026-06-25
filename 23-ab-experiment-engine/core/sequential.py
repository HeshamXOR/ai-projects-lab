"""Sequential testing: SPRT and alpha-spending (from scratch).

WHY THIS MODULE EXISTS
----------------------
In classic fixed-horizon testing you decide the sample size in advance and only
look at the result once. In practice, experimenters "peek" repeatedly and stop
as soon as they see significance -- which inflates the false-positive rate
dramatically (peeking 10 times at alpha=0.05 gives a real type-I rate well above
20%). Sequential methods are designed to let you monitor continuously while
controlling the overall error rate.

Two complementary approaches are implemented:

1. **SPRT (Wald's Sequential Probability Ratio Test).** We accumulate the
   log-likelihood ratio of the data under H1 vs H0. Two horizontal boundaries
   derived from ``alpha`` and ``beta`` decide the test::

       A = log( beta / (1 - alpha) )      (lower boundary -> accept H0)
       B = log( (1 - beta) / alpha )      (upper boundary -> reject H0)

   While ``A < LLR < B`` we keep sampling. For Bernoulli data the per-sample
   log-likelihood-ratio increment has a closed form, implemented in
   :func:`sprt_bernoulli`.

2. **Alpha-spending (group-sequential).** Pocock and O'Brien-Fleming spending
   functions allocate the total ``alpha`` budget across ``K`` planned looks as a
   function of the information fraction ``t = n / n_max``. At each look we
   compute the boundary z-value the cumulative spend implies and compare the
   observed z. O'Brien-Fleming is conservative early and nearly full-alpha at the
   end; Pocock spends evenly. Implemented in :func:`alpha_spending_boundary` and
   driven by :func:`group_sequential_decision`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .distributions import normal_cdf, normal_ppf

__all__ = [
    "SequentialDecision",
    "SPRTResult",
    "sprt_bernoulli",
    "SpendingFunction",
    "alpha_spending_boundary",
    "GroupSequentialResult",
    "group_sequential_decision",
]


class SequentialDecision(str, Enum):
    """The three possible verdicts at any peek."""

    CONTINUE = "continue"
    STOP_REJECT_H0 = "stop_reject_h0"  # treatment effect detected -> ship
    STOP_ACCEPT_H0 = "stop_accept_h0"  # no effect, stop for futility


@dataclass(frozen=True)
class SPRTResult:
    """Outcome of evaluating Wald's SPRT on the data so far."""

    decision: SequentialDecision
    log_likelihood_ratio: float
    lower_boundary: float
    upper_boundary: float
    p0: float
    p1: float
    n_a: int
    n_b: int

    def as_dict(self) -> dict:
        return {
            "method": "sprt",
            "decision": self.decision.value,
            "log_likelihood_ratio": self.log_likelihood_ratio,
            "lower_boundary": self.lower_boundary,
            "upper_boundary": self.upper_boundary,
            "p0": self.p0,
            "p1": self.p1,
            "n_a": self.n_a,
            "n_b": self.n_b,
        }


def sprt_bernoulli(
    conversions_a: int,
    n_a: int,
    conversions_b: int,
    n_b: int,
    *,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> SPRTResult:
    """Wald SPRT for a Bernoulli treatment effect.

    We test ``H0: treatment rate == p0`` against ``H1: treatment rate == p1``
    using the treatment arm's conversions, while the control arm informs the
    operating points (``p0`` is typically the observed control rate, ``p1`` the
    control rate plus the minimum detectable effect).

    The cumulative log-likelihood ratio for ``k`` successes in ``n`` Bernoulli
    trials is::

        LLR = k * log(p1/p0) + (n - k) * log((1-p1)/(1-p0))

    Decision rule:
        LLR >= B  -> reject H0 (effect detected)
        LLR <= A  -> accept H0 (futility)
        otherwise -> continue sampling
    """
    if not 0.0 < p0 < 1.0 or not 0.0 < p1 < 1.0:
        raise ValueError("p0 and p1 must be in (0, 1)")
    if p0 == p1:
        raise ValueError("p0 and p1 must differ")
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must be in (0, 1)")

    k = conversions_b
    n = n_b
    llr = k * math.log(p1 / p0) + (n - k) * math.log((1.0 - p1) / (1.0 - p0))

    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)

    if llr >= upper:
        decision = SequentialDecision.STOP_REJECT_H0
    elif llr <= lower:
        decision = SequentialDecision.STOP_ACCEPT_H0
    else:
        decision = SequentialDecision.CONTINUE

    return SPRTResult(
        decision=decision,
        log_likelihood_ratio=llr,
        lower_boundary=lower,
        upper_boundary=upper,
        p0=p0,
        p1=p1,
        n_a=n_a,
        n_b=n_b,
    )


class SpendingFunction(str, Enum):
    """Supported alpha-spending functions."""

    POCOCK = "pocock"
    OBRIEN_FLEMING = "obrien_fleming"


def alpha_spending_boundary(
    information_fraction: float,
    *,
    alpha: float = 0.05,
    spending: SpendingFunction = SpendingFunction.OBRIEN_FLEMING,
    two_sided: bool = True,
) -> float:
    """Critical |z| boundary at a given information fraction ``t`` in (0, 1].

    Uses the Lan-DeMets spending-function representation:

    * Pocock:           alpha*(t) = alpha * ln(1 + (e - 1) * t)
    * O'Brien-Fleming:  alpha*(t) = 2 * (1 - Phi( z_{1-alpha/2} / sqrt(t) ))

    The returned value is the z critical value such that crossing it at this look
    spends the cumulative budget ``alpha*(t)``. For O'Brien-Fleming the boundary
    is large early (hard to stop) and approaches ``z_{1-alpha/2}`` at ``t = 1``.
    """
    if not 0.0 < information_fraction <= 1.0:
        raise ValueError("information_fraction must be in (0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    t = information_fraction
    if spending is SpendingFunction.OBRIEN_FLEMING:
        z_full = normal_ppf(1.0 - alpha / 2.0) if two_sided else normal_ppf(1.0 - alpha)
        spent = 2.0 * (1.0 - normal_cdf(z_full / math.sqrt(t)))
    else:  # POCOCK
        spent = alpha * math.log(1.0 + (math.e - 1.0) * t)

    spent = min(spent, alpha)  # never spend more than the budget
    spent = max(spent, 1e-12)  # guard log/quantile domain

    if two_sided:
        return normal_ppf(1.0 - spent / 2.0)
    return normal_ppf(1.0 - spent)


@dataclass(frozen=True)
class GroupSequentialResult:
    """Outcome of a group-sequential peek using an alpha-spending boundary."""

    decision: SequentialDecision
    observed_z: float
    boundary_z: float
    information_fraction: float
    spending: str
    alpha: float

    def as_dict(self) -> dict:
        return {
            "method": "alpha_spending",
            "spending_function": self.spending,
            "decision": self.decision.value,
            "observed_z": self.observed_z,
            "boundary_z": self.boundary_z,
            "information_fraction": self.information_fraction,
            "alpha": self.alpha,
        }


def group_sequential_decision(
    observed_z: float,
    n_current_per_arm: float,
    n_planned_per_arm: float,
    *,
    alpha: float = 0.05,
    spending: SpendingFunction = SpendingFunction.OBRIEN_FLEMING,
    two_sided: bool = True,
) -> GroupSequentialResult:
    """Decide continue / stop given the current z statistic and information fraction.

    ``information_fraction = n_current / n_planned`` (capped at 1.0). If the
    observed ``|z|`` exceeds the spending boundary we stop and reject H0. At full
    information (``t >= 1``) a sub-boundary z means we stop for futility;
    otherwise we continue collecting data.
    """
    if n_planned_per_arm <= 0:
        raise ValueError("n_planned_per_arm must be positive")
    t = min(max(n_current_per_arm / n_planned_per_arm, 1e-9), 1.0)

    boundary = alpha_spending_boundary(
        t, alpha=alpha, spending=spending, two_sided=two_sided
    )

    if abs(observed_z) >= boundary:
        decision = SequentialDecision.STOP_REJECT_H0
    elif t >= 1.0:
        decision = SequentialDecision.STOP_ACCEPT_H0
    else:
        decision = SequentialDecision.CONTINUE

    return GroupSequentialResult(
        decision=decision,
        observed_z=observed_z,
        boundary_z=boundary,
        information_fraction=t,
        spending=spending.value,
        alpha=alpha,
    )
