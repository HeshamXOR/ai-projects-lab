"""Tests for the from-scratch distributions (core.distributions).

All expected values are textbook constants, cited inline.
"""

from __future__ import annotations

import math

from core.distributions import (
    erf,
    normal_cdf,
    normal_ppf,
    students_t_cdf,
    students_t_sf,
)
from core.distributions import _erf_series  # exercise the hand-rolled path


def test_normal_cdf_at_zero_is_half():
    # Phi(0) = 0.5 exactly.
    assert abs(normal_cdf(0.0) - 0.5) < 1e-12


def test_normal_cdf_at_1_96_is_0_975():
    # Phi(1.96) = 0.9750021... (standard z table). Tolerance 1e-3 as specified.
    assert abs(normal_cdf(1.96) - 0.975) < 1e-3


def test_normal_cdf_symmetry():
    # Phi(-x) = 1 - Phi(x).
    for x in (0.3, 1.0, 2.5, 3.1):
        assert abs(normal_cdf(-x) - (1.0 - normal_cdf(x))) < 1e-12


def test_inverse_normal_at_0_975_is_1_96():
    # The canonical two-sided 95% critical value: Phi^{-1}(0.975) = 1.959963985.
    z = normal_ppf(0.975)
    assert abs(z - 1.959963985) < 1e-3


def test_inverse_normal_round_trip():
    # ppf(cdf(x)) == x to high precision (Halley-refined).
    for x in (-2.0, -0.5, 0.25, 1.5, 2.8):
        assert abs(normal_ppf(normal_cdf(x)) - x) < 1e-8


def test_inverse_normal_known_quantiles():
    # 90% one-sided -> 1.2815515; 99% one-sided -> 2.3263479 (z tables).
    assert abs(normal_ppf(0.90) - 1.2815515) < 1e-4
    assert abs(normal_ppf(0.99) - 2.3263479) < 1e-4


def test_hand_rolled_erf_matches_math_erf():
    # The A&S series is good to ~1.5e-7.
    for x in (-2.0, -0.7, 0.0, 0.7, 1.5, 3.0):
        assert abs(_erf_series(x) - math.erf(x)) < 2e-7


def test_erf_endpoints():
    assert abs(erf(0.0)) < 1e-12
    assert abs(erf(5.0) - 1.0) < 1e-6


def test_students_t_cdf_at_zero_is_half():
    for df in (1.0, 5.0, 30.0):
        assert abs(students_t_cdf(0.0, df) - 0.5) < 1e-12


def test_students_t_cdf_known_critical_values():
    # Textbook two-sided 95% t critical values (upper tail prob 0.025):
    #   df=10 -> 2.228, df=20 -> 2.086, df=1 (Cauchy) -> 12.706.
    # So F_t(t_crit, df) should be ~0.975.
    assert abs(students_t_cdf(2.228, 10.0) - 0.975) < 1e-3
    assert abs(students_t_cdf(2.086, 20.0) - 0.975) < 1e-3
    assert abs(students_t_cdf(12.706, 1.0) - 0.975) < 1e-3


def test_students_t_approaches_normal_for_large_df():
    # As df -> inf the t-distribution converges to the standard normal.
    for x in (0.5, 1.0, 1.96):
        assert abs(students_t_cdf(x, 5000.0) - normal_cdf(x)) < 1e-3


def test_students_t_sf_complements_cdf():
    for t in (-1.5, 0.4, 2.2):
        assert abs(students_t_sf(t, 8.0) - (1.0 - students_t_cdf(t, 8.0))) < 1e-12
