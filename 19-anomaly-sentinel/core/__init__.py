"""anomaly-sentinel core: drift tests, online detectors, and the rule engine.

All algorithms are implemented from scratch in NumPy (no SciPy / scikit-learn):

* :mod:`core.drift`     — PSI and two-sample KS test.
* :mod:`core.detector`  — streaming EWMA/z-score + simplified Isolation Forest.
* :mod:`core.rules`     — severity-graded alert rule engine with cooldown.
"""

from .drift import (
    KSResult,
    PSIResult,
    drift_report,
    ks_two_sample,
    population_stability_index,
)
from .detector import (
    EWMADetector,
    IsolationForestScorer,
    ScorePoint,
    expected_path_length,
)
from .rules import Alert, Rule, RuleEngine, Severity, Signal, default_rules

__all__ = [
    # drift
    "population_stability_index",
    "ks_two_sample",
    "drift_report",
    "PSIResult",
    "KSResult",
    # detectors
    "EWMADetector",
    "IsolationForestScorer",
    "ScorePoint",
    "expected_path_length",
    # rules
    "RuleEngine",
    "Rule",
    "Signal",
    "Alert",
    "Severity",
    "default_rules",
]
