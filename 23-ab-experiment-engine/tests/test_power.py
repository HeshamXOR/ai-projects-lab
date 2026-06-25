"""Tests for power/sample-size (core.power) against known results."""

from __future__ import annotations

from core.power import (
    power_for_sample_size,
    sample_size_for_mean,
    sample_size_two_proportion,
)


def test_sample_size_known_result():
    """Known two-proportion sample size.

    Baseline p = 0.10, detect lift to 0.12 (mde = 0.02), alpha=0.05, power=0.80.

    z_{0.975} = 1.959964, z_{0.80} = 0.841621
    p_bar = 0.11
    num = 1.959964*sqrt(2*0.11*0.89) + 0.841621*sqrt(0.1*0.9 + 0.12*0.88)
        = 1.959964*sqrt(0.1958) + 0.841621*sqrt(0.09 + 0.1056)
        = 1.959964*0.442493 + 0.841621*0.442267
        = 0.867257 + 0.372219 = 1.239476
    n = 1.239476^2 / 0.02^2 = 1.536300 / 0.0004 = 3840.75 -> 3841 per arm.

    Standard references (e.g. Evan Miller's calculator) give ~3,843 per arm; the
    exact integer depends on rounding of the z-values, so allow a small window.
    """
    r = sample_size_two_proportion(0.10, 0.02, alpha=0.05, power=0.80)
    assert 3830 <= r.required_n_per_arm <= 3855
    assert abs(r.z_alpha - 1.959964) < 1e-4
    assert abs(r.z_beta - 0.841621) < 1e-4
    assert abs(r.treatment_rate - 0.12) < 1e-12


def test_sample_size_larger_effect_needs_fewer_samples():
    small = sample_size_two_proportion(0.10, 0.02).required_n_per_arm
    large = sample_size_two_proportion(0.10, 0.05).required_n_per_arm
    assert large < small


def test_power_for_sample_size_round_trips():
    # Plug the required n back in -> achieved power should be ~ the target 0.80.
    r = sample_size_two_proportion(0.10, 0.02, alpha=0.05, power=0.80)
    achieved = power_for_sample_size(0.10, 0.02, r.required_n_per_arm, alpha=0.05)
    assert abs(achieved - 0.80) < 0.01


def test_power_increases_with_sample_size():
    p_small = power_for_sample_size(0.10, 0.02, 500)
    p_large = power_for_sample_size(0.10, 0.02, 5000)
    assert p_large > p_small
    assert 0.0 < p_small < p_large < 1.0


def test_sample_size_for_mean_known_result():
    """Continuous-metric sample size.

    sigma=1.0, mde=0.5, alpha=0.05, power=0.80.
    n = 2 * 1 * (1.959964 + 0.841621)^2 / 0.25
      = 2 * (2.801585)^2 / 0.25
      = 2 * 7.848877 / 0.25 = 15.697754 / 0.25 = 62.79 -> 63 per arm.
    """
    n = sample_size_for_mean(1.0, 0.5, alpha=0.05, power=0.80)
    assert 62 <= n <= 64
