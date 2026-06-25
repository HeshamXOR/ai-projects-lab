"""Tests for sequential testing (core.sequential) on constructed streams."""

from __future__ import annotations

import math

from core.sequential import (
    SequentialDecision,
    SpendingFunction,
    alpha_spending_boundary,
    group_sequential_decision,
    sprt_bernoulli,
)


def test_sprt_continues_early():
    # With very little data the SPRT should not yet decide.
    r = sprt_bernoulli(5, 50, 6, 50, p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
    assert r.lower_boundary < r.log_likelihood_ratio < r.upper_boundary
    assert r.decision is SequentialDecision.CONTINUE


def test_sprt_rejects_h0_on_strong_treatment():
    # Treatment converting at ~16% over a large sample vs p0=0.10 should pile up
    # enough evidence to cross the upper boundary and reject H0 (detect effect).
    r = sprt_bernoulli(100, 1000, 320, 2000, p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
    assert r.log_likelihood_ratio > r.upper_boundary
    assert r.decision is SequentialDecision.STOP_REJECT_H0


def test_sprt_accepts_h0_on_null_treatment():
    # Treatment essentially at p0 should drift toward the lower boundary -> futility.
    r = sprt_bernoulli(100, 1000, 198, 2000, p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
    assert r.log_likelihood_ratio < r.lower_boundary
    assert r.decision is SequentialDecision.STOP_ACCEPT_H0


def test_sprt_boundaries_have_expected_signs():
    # A = log(beta/(1-alpha)) < 0 < log((1-beta)/alpha) = B.
    r = sprt_bernoulli(1, 10, 1, 10, p0=0.10, p1=0.20, alpha=0.05, beta=0.20)
    assert r.lower_boundary < 0.0 < r.upper_boundary
    assert abs(r.lower_boundary - math.log(0.20 / 0.95)) < 1e-12
    assert abs(r.upper_boundary - math.log(0.80 / 0.05)) < 1e-12


def test_obrien_fleming_boundary_is_conservative_early():
    # O'Brien-Fleming spends little alpha early: boundary at t=0.25 should be far
    # larger than the fixed 1.96, and approach 1.96 at t=1.
    early = alpha_spending_boundary(0.25, alpha=0.05, spending=SpendingFunction.OBRIEN_FLEMING)
    late = alpha_spending_boundary(1.0, alpha=0.05, spending=SpendingFunction.OBRIEN_FLEMING)
    assert early > late
    assert early > 2.5
    assert abs(late - 1.959964) < 1e-2


def test_pocock_boundary_roughly_constant():
    # Pocock spends evenly, so its boundary varies far less across looks than OBF.
    p_early = alpha_spending_boundary(0.25, spending=SpendingFunction.POCOCK)
    p_late = alpha_spending_boundary(1.0, spending=SpendingFunction.POCOCK)
    obf_early = alpha_spending_boundary(0.25, spending=SpendingFunction.OBRIEN_FLEMING)
    # Pocock's early-to-late spread is much smaller than OBF's.
    assert (p_early - p_late) < (obf_early - p_late)


def test_group_sequential_continue_when_underpowered_and_below_boundary():
    # Small z at 30% information -> keep running.
    gs = group_sequential_decision(1.5, 300, 1000, alpha=0.05)
    assert gs.decision is SequentialDecision.CONTINUE
    assert gs.information_fraction == 0.3


def test_group_sequential_stops_reject_when_z_crosses_boundary():
    # A huge z at any information fraction crosses even the conservative boundary.
    gs = group_sequential_decision(6.0, 300, 1000, alpha=0.05)
    assert gs.decision is SequentialDecision.STOP_REJECT_H0
    assert abs(gs.observed_z) >= gs.boundary_z


def test_group_sequential_futility_at_full_information():
    # At full information with a small z -> stop for futility.
    gs = group_sequential_decision(0.5, 1000, 1000, alpha=0.05)
    assert gs.decision is SequentialDecision.STOP_ACCEPT_H0
    assert gs.information_fraction == 1.0
