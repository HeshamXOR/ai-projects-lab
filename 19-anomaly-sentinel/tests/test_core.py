"""Proofs for the from-scratch drift tests, detectors, and rule engine."""

import numpy as np

from core.detector import EWMADetector, IsolationForestScorer, expected_path_length
from core.drift import ks_two_sample, population_stability_index
from core.rules import RuleEngine, Severity, Signal


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #
def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, size=5000)
    act = rng.normal(0, 1, size=5000)
    res = population_stability_index(ref, act, n_bins=10)
    assert res.psi < 0.1
    assert res.drift is False


def test_psi_large_for_injected_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, size=5000)
    act = rng.normal(3, 1, size=5000)  # big mean shift
    res = population_stability_index(ref, act, n_bins=10)
    assert res.psi > 0.2
    assert res.drift is True


# --------------------------------------------------------------------------- #
# KS
# --------------------------------------------------------------------------- #
def test_ks_detects_different_distributions():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(2, 1, size=2000)
    res = ks_two_sample(a, b)
    assert res.p_value < 0.05
    assert res.drift is True


def test_ks_accepts_identical_distributions():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(0, 1, size=2000)
    res = ks_two_sample(a, b)
    assert res.p_value > 0.05
    assert res.drift is False


# --------------------------------------------------------------------------- #
# EWMA / z-score detector
# --------------------------------------------------------------------------- #
def test_ewma_flags_injected_spike():
    rng = np.random.default_rng(4)
    stable = rng.normal(10.0, 0.5, size=200)
    det = EWMADetector(alpha=0.05, z_threshold=4.0, warmup=20)
    points = det.update_many(stable)
    # No spurious flags on the clean stable stream.
    assert not any(p.is_anomaly for p in points)

    # Now inject an obvious spike — it should be flagged.
    spike = det.update(50.0)
    assert spike.is_anomaly is True
    assert abs(spike.z) > 4.0


def test_ewma_does_not_flag_pure_noise():
    rng = np.random.default_rng(5)
    noise = rng.normal(0.0, 1.0, size=500)
    det = EWMADetector(alpha=0.1, z_threshold=4.0, warmup=20)
    points = det.update_many(noise)
    flagged = sum(p.is_anomaly for p in points)
    # A handful of >4 sigma events at most over 500 gaussian samples.
    assert flagged <= 3


# --------------------------------------------------------------------------- #
# Isolation-style scorer
# --------------------------------------------------------------------------- #
def test_isolation_scores_outlier_higher_than_inliers():
    rng = np.random.default_rng(6)
    inliers = rng.normal(0.0, 1.0, size=300).reshape(-1, 1)
    scorer = IsolationForestScorer(n_trees=100, sample_size=128, random_state=7).fit(inliers)

    inlier_scores = scorer.score(inliers)
    outlier_score = scorer.score(np.array([[25.0]]))[0]

    assert outlier_score > np.mean(inlier_scores)
    assert outlier_score > 0.5  # canonical anomaly threshold


def test_expected_path_length_monotonic():
    assert expected_path_length(2) < expected_path_length(100)
    assert expected_path_length(1) == 1.0


# --------------------------------------------------------------------------- #
# Rule engine
# --------------------------------------------------------------------------- #
def test_rule_engine_critical_on_drift_and_anomaly():
    engine = RuleEngine(clock=lambda: 1000.0)
    sig = Signal(metric="latency", value=99.0, drift=True, anomaly_score=0.9, z=5.0)
    alert = engine.evaluate(sig)
    assert alert is not None
    assert alert.severity == Severity.CRITICAL
    assert alert.rule == "drift_and_anomaly"


def test_rule_engine_respects_cooldown():
    t = {"now": 0.0}
    engine = RuleEngine(cooldown_seconds=60.0, clock=lambda: t["now"])

    sig = Signal(metric="m", value=10.0, z=6.0)  # WARNING point anomaly
    first = engine.evaluate(sig)
    assert first is not None

    # Same metric+rule within cooldown -> suppressed.
    t["now"] = 30.0
    second = engine.evaluate(sig)
    assert second is None

    # After the cooldown window -> fires again.
    t["now"] = 120.0
    third = engine.evaluate(sig)
    assert third is not None


def test_rule_engine_no_match_returns_none():
    engine = RuleEngine(clock=lambda: 0.0)
    sig = Signal(metric="m", value=1.0, z=0.1, anomaly_score=0.1, drift=False)
    assert engine.evaluate(sig) is None
