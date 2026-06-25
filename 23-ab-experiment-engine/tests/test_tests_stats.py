"""Tests for the hypothesis tests (core.tests_stats) against textbook values."""

from __future__ import annotations

import math

from core.tests_stats import two_proportion_z_test, welch_t_test


def test_two_proportion_z_known_example():
    """Known worked example.

    Control: 1000 trials, 100 conversions -> p_a = 0.10
    Treatment: 1000 trials, 130 conversions -> p_b = 0.13

    Pooled p = 230/2000 = 0.115
    SE_pooled = sqrt(0.115 * 0.885 * (1/1000 + 1/1000))
              = sqrt(0.101775 * 0.002) = sqrt(0.00020355) = 0.0142671
    z = (0.13 - 0.10) / 0.0142671 = 0.03 / 0.0142671 = 2.1027
    two-sided p = 2 * (1 - Phi(2.1027)) ~= 0.0355
    """
    r = two_proportion_z_test(100, 1000, 130, 1000, alpha=0.05)
    assert abs(r.estimate - 0.03) < 1e-12
    assert abs(r.statistic - 2.1027) < 1e-3
    assert abs(r.p_value - 0.0355) < 2e-3
    assert r.significant is True
    # Relative lift = (0.13 - 0.10) / 0.10 = 0.30.
    assert abs(r.relative_lift - 0.30) < 1e-9


def test_two_proportion_z_no_effect():
    # Identical rates -> z = 0, p = 1, not significant.
    r = two_proportion_z_test(150, 1000, 150, 1000)
    assert abs(r.statistic) < 1e-9
    assert abs(r.p_value - 1.0) < 1e-9
    assert r.significant is False


def test_two_proportion_ci_brackets_estimate():
    r = two_proportion_z_test(100, 1000, 130, 1000, alpha=0.05)
    assert r.ci_low < r.estimate < r.ci_high
    # 95% CI half-width = 1.96 * SE_unpooled.
    # SE_unpooled = sqrt(0.1*0.9/1000 + 0.13*0.87/1000) = sqrt(9e-5 + 1.131e-4)
    #             = sqrt(2.031e-4) = 0.014252; margin ~= 0.027934.
    margin = (r.ci_high - r.ci_low) / 2.0
    assert abs(margin - 0.027934) < 1e-3


def test_welch_t_textbook_example():
    """Welch's t-test textbook dataset (the classic Ruxton/Welch example).

    Sample A (control): mean=20.0, var=16.0, n=10
    Sample B (treatment): mean=22.0, var=25.0, n=12

    se = sqrt(16/10 + 25/12) = sqrt(1.6 + 2.083333) = sqrt(3.683333) = 1.91920
    t  = (22 - 20) / 1.91920 = 1.04210
    df = (1.6 + 2.083333)^2 / ( 1.6^2/9 + 2.083333^2/11 )
       = 13.567 / ( 2.56/9 + 4.34028/11 )
       = 13.567 / (0.284444 + 0.394571)
       = 13.567 / 0.679015 = 19.980
    """
    r = welch_t_test(
        mean_a=20.0, var_a=16.0, n_a=10,
        mean_b=22.0, var_b=25.0, n_b=12,
        alpha=0.05,
    )
    assert abs(r.estimate - 2.0) < 1e-12
    assert abs(r.statistic - 1.04210) < 1e-3
    assert abs(r.df - 19.980) < 5e-2


def test_welch_t_from_raw_samples():
    a = [5.1, 4.9, 5.0, 5.2, 4.8, 5.05, 4.95]
    b = [5.4, 5.6, 5.5, 5.45, 5.65, 5.35, 5.5]
    r = welch_t_test(a, b, alpha=0.05)
    # Treatment mean clearly above control -> positive estimate, significant.
    assert r.estimate > 0
    assert r.significant is True
    assert r.df > 0


def test_welch_p_value_matches_t_cdf():
    # For t=1.04210, df=19.98 the two-sided p should be ~0.310.
    r = welch_t_test(
        mean_a=20.0, var_a=16.0, n_a=10,
        mean_b=22.0, var_b=25.0, n_b=12,
    )
    assert abs(r.p_value - 0.310) < 1e-2
    assert r.significant is False
