"""Core statistics engine for the A/B experiment platform.

Every statistical method here is implemented from scratch in pure Python/NumPy.
No SciPy is used: the normal CDF, the inverse normal CDF (quantile), and the
Student-t CDF are all hand-rolled in :mod:`core.distributions` and underpin the
hypothesis tests, power calculations, and sequential boundaries.

Public surface
--------------
* :mod:`core.distributions` -- erf, normal cdf/pdf/ppf, Student-t cdf/sf.
* :mod:`core.tests_stats`   -- two-proportion z-test, Welch's t-test, CIs.
* :mod:`core.power`         -- power & sample-size calculations.
* :mod:`core.sequential`    -- SPRT and alpha-spending group-sequential tests.
* :mod:`core.cuped`         -- CUPED variance reduction.
"""

from __future__ import annotations

from .cuped import CupedResult, apply_cuped, compute_theta
from .distributions import (
    erf,
    normal_cdf,
    normal_pdf,
    normal_ppf,
    normal_sf,
    students_t_cdf,
    students_t_sf,
)
from .power import (
    PowerResult,
    power_for_sample_size,
    sample_size_for_mean,
    sample_size_two_proportion,
)
from .sequential import (
    GroupSequentialResult,
    SPRTResult,
    SequentialDecision,
    SpendingFunction,
    alpha_spending_boundary,
    group_sequential_decision,
    sprt_bernoulli,
)
from .tests_stats import TestResult, lift, two_proportion_z_test, welch_t_test

__all__ = [
    # distributions
    "erf",
    "normal_cdf",
    "normal_pdf",
    "normal_sf",
    "normal_ppf",
    "students_t_cdf",
    "students_t_sf",
    # tests
    "TestResult",
    "two_proportion_z_test",
    "welch_t_test",
    "lift",
    # power
    "PowerResult",
    "sample_size_two_proportion",
    "power_for_sample_size",
    "sample_size_for_mean",
    # sequential
    "SequentialDecision",
    "SPRTResult",
    "sprt_bernoulli",
    "SpendingFunction",
    "alpha_spending_boundary",
    "GroupSequentialResult",
    "group_sequential_decision",
    # cuped
    "CupedResult",
    "apply_cuped",
    "compute_theta",
]
