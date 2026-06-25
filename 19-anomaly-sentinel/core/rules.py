"""Alerting rule engine with severity, dedup and cooldown — from scratch.

The detectors and drift tests emit raw *signals* (a z-score, an anomaly score, a
drift verdict, a threshold breach). This module turns those into actionable
:class:`Alert` objects through a small, configurable rule engine:

* **Rules** map a predicate over a :class:`Signal` to a :class:`Severity`.
* The first matching rule (highest severity wins) decides the alert level.
* **Cooldown / dedup** suppresses repeat alerts for the same ``(metric, rule)``
  key within a configurable window, so a sustained anomaly fires once rather
  than on every sample.

No external dependencies — plain dataclasses and a deterministic clock hook for
testability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple


class Severity(IntEnum):
    """Ordered alert severity levels (higher = more urgent)."""

    INFO = 10
    WARNING = 20
    CRITICAL = 30

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


@dataclass
class Signal:
    """The bundle of monitoring signals for a single metric observation."""

    metric: str
    value: float = 0.0
    z: float = 0.0
    anomaly_score: float = 0.0
    drift: bool = False
    psi: float = 0.0
    ks_p_value: float = 1.0
    threshold_breached: bool = False
    timestamp: float = 0.0


@dataclass
class Rule:
    """A single alerting rule.

    Parameters
    ----------
    name:
        Stable identifier (also the dedup key component).
    severity:
        Severity assigned when the rule matches.
    predicate:
        Callable ``Signal -> bool`` deciding whether the rule fires.
    message:
        Human-readable template; ``{metric}`` and ``{value}`` are filled in.
    """

    name: str
    severity: Severity
    predicate: Callable[[Signal], bool]
    message: str = "{metric}: rule fired (value={value})"

    def matches(self, signal: Signal) -> bool:
        return bool(self.predicate(signal))

    def render(self, signal: Signal) -> str:
        return self.message.format(metric=signal.metric, value=round(signal.value, 4))


@dataclass
class Alert:
    """A structured, emitted alert."""

    metric: str
    rule: str
    severity: Severity
    message: str
    timestamp: float
    signal: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "rule": self.rule,
            "severity": str(self.severity),
            "severity_level": int(self.severity),
            "message": self.message,
            "timestamp": self.timestamp,
            "signal": self.signal,
        }


def default_rules(
    z_threshold: float = 3.0,
    anomaly_threshold: float = 0.6,
) -> List[Rule]:
    """A sensible default rule set.

    The combination rule (drift *and* anomaly) is intentionally listed first and
    given CRITICAL severity: correlated drift + point anomalies are the strongest
    "something is really wrong" signal in model monitoring.
    """
    return [
        Rule(
            name="drift_and_anomaly",
            severity=Severity.CRITICAL,
            predicate=lambda s: s.drift and s.anomaly_score >= anomaly_threshold,
            message="{metric}: CRITICAL — distribution drift AND point anomaly (value={value})",
        ),
        Rule(
            name="hard_threshold",
            severity=Severity.CRITICAL,
            predicate=lambda s: s.threshold_breached,
            message="{metric}: CRITICAL — hard threshold breached (value={value})",
        ),
        Rule(
            name="distribution_drift",
            severity=Severity.WARNING,
            predicate=lambda s: s.drift,
            message="{metric}: WARNING — distribution drift detected",
        ),
        Rule(
            name="point_anomaly",
            severity=Severity.WARNING,
            predicate=lambda s: abs(s.z) > z_threshold or s.anomaly_score >= anomaly_threshold,
            message="{metric}: WARNING — anomalous value (z={value})",
        ),
    ]


@dataclass
class RuleEngine:
    """Evaluate signals against rules, with severity, dedup and cooldown.

    Parameters
    ----------
    rules:
        Ordered rule list; the highest-severity match wins for a given signal.
    cooldown_seconds:
        Minimum time between two alerts sharing the same ``(metric, rule)`` key.
    clock:
        Injectable time source (defaults to :func:`time.time`) — tests pass a
        deterministic counter to exercise cooldown logic.
    """

    rules: List[Rule] = field(default_factory=default_rules)
    cooldown_seconds: float = 60.0
    clock: Callable[[], float] = time.time

    _last_fired: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def evaluate(self, signal: Signal) -> Optional[Alert]:
        """Evaluate one signal; return an :class:`Alert` or ``None``.

        Returns ``None`` when no rule matches, or when the best-matching rule is
        still within its cooldown window for this metric (dedup).
        """
        matched = [r for r in self.rules if r.matches(signal)]
        if not matched:
            return None

        # Highest severity wins; ties broken by rule order (stable sort).
        best = max(matched, key=lambda r: r.severity)
        now = signal.timestamp or self.clock()
        key = (signal.metric, best.name)

        last = self._last_fired.get(key)
        if last is not None and (now - last) < self.cooldown_seconds:
            return None  # suppressed by cooldown / dedup

        self._last_fired[key] = now
        return Alert(
            metric=signal.metric,
            rule=best.name,
            severity=best.severity,
            message=best.render(signal),
            timestamp=now,
            signal={
                "value": signal.value,
                "z": signal.z,
                "anomaly_score": signal.anomaly_score,
                "psi": signal.psi,
                "ks_p_value": signal.ks_p_value,
                "drift": float(signal.drift),
            },
        )

    def reset(self) -> None:
        """Clear all cooldown state (e.g. between independent runs)."""
        self._last_fired.clear()
