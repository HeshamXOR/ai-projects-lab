"""Severity scoring: combine rule hits and classifier probability.

Produces a per-category score in ``[0, 1]``, an overall score, and a decision
(``allow`` / ``flag`` / ``block``) from configurable thresholds.

The category score blends two signals:

* The strongest rule severity in the category (rules express hard knowledge:
  a Luhn-valid card or a credit-card threat is intrinsically severe).
* For toxicity / self-harm, the Naive Bayes ``P(toxic)`` is folded in so the
  classifier can lift borderline text that dodges every keyword.

Combination uses a "noisy-OR"-style rule so that multiple independent signals
accumulate toward 1.0 without ever exceeding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Sequence

from .policy import RuleHit
from .rules import Category


class Decision(str, Enum):
    """Final moderation decision."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


# Default decision thresholds on the overall score.
DEFAULT_FLAG_THRESHOLD = 0.35
DEFAULT_BLOCK_THRESHOLD = 0.70

# How strongly the classifier's P(toxic) feeds the toxicity score.
DEFAULT_CLASSIFIER_WEIGHT = 0.6

# The classifier only contributes once it leans toxic past this confidence.
# Below it (near the 0.5 prior, or clean-leaning) it adds nothing, so a small
# bundled model cannot push obviously-clean text into FLAG on noise alone.
DEFAULT_CLASSIFIER_FLOOR = 0.6


def _classifier_signal(
    toxic_probability: float,
    floor: float,
    weight: float,
) -> float:
    """Map ``P(toxic)`` to a bounded toxicity contribution.

    Probabilities below ``floor`` contribute nothing. Above it, the excess is
    rescaled to ``[0, 1]`` and multiplied by ``weight`` so confident toxic
    predictions lift the score while uncertain ones stay silent.
    """
    if toxic_probability <= floor:
        return 0.0
    rescaled = (toxic_probability - floor) / (1.0 - floor)
    return rescaled * weight


def _noisy_or(values: Sequence[float]) -> float:
    """Combine independent probabilities via ``1 - prod(1 - v)``."""
    product = 1.0
    for v in values:
        product *= (1.0 - max(0.0, min(1.0, v)))
    return 1.0 - product


@dataclass
class ScoreResult:
    """Outcome of scoring a single text.

    Attributes:
        decision: ``allow`` / ``flag`` / ``block``.
        overall: Overall severity score in ``[0, 1]``.
        category_scores: Per-category severity in ``[0, 1]``.
        toxic_probability: Classifier ``P(toxic)`` (the toxic-class proba).
        flag_threshold: Threshold used for the FLAG cutoff.
        block_threshold: Threshold used for the BLOCK cutoff.
    """

    decision: Decision
    overall: float
    category_scores: Dict[str, float]
    toxic_probability: float
    flag_threshold: float
    block_threshold: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "decision": self.decision.value,
            "overall": round(self.overall, 4),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "toxic_probability": round(self.toxic_probability, 4),
            "thresholds": {
                "flag": self.flag_threshold,
                "block": self.block_threshold,
            },
        }


def score_text(
    hits: Sequence[RuleHit],
    toxic_probability: float,
    *,
    classifier_weight: float = DEFAULT_CLASSIFIER_WEIGHT,
    classifier_floor: float = DEFAULT_CLASSIFIER_FLOOR,
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
) -> ScoreResult:
    """Combine rule hits and classifier probability into a decision.

    Args:
        hits: Rule hits from the policy engine.
        toxic_probability: ``P(toxic)`` from the Naive Bayes classifier.
        classifier_weight: Weight applied to the classifier signal when it is
            folded into the toxicity category.
        classifier_floor: Confidence below which the classifier contributes
            nothing (suppresses small-data noise near the 0.5 prior).
        flag_threshold: Overall score at/above which to FLAG.
        block_threshold: Overall score at/above which to BLOCK.

    Returns:
        A populated :class:`ScoreResult`.
    """
    if not 0.0 <= flag_threshold <= block_threshold <= 1.0:
        raise ValueError("require 0 <= flag_threshold <= block_threshold <= 1")
    toxic_probability = max(0.0, min(1.0, toxic_probability))

    # Collect rule severities per category.
    per_cat_sevs: Dict[str, List[float]] = {c.value: [] for c in Category}
    for hit in hits:
        per_cat_sevs[hit.category.value].append(hit.severity)

    # Inject the classifier signal into toxicity (floored + rescaled).
    per_cat_sevs[Category.TOXICITY.value].append(
        _classifier_signal(toxic_probability, classifier_floor, classifier_weight)
    )

    category_scores = {
        cat: _noisy_or(sevs) for cat, sevs in per_cat_sevs.items()
    }

    # Overall is the strongest category -- one severe category should block
    # regardless of the others, but we add a small noisy-OR bump so several
    # moderate categories together can also escalate.
    strongest = max(category_scores.values()) if category_scores else 0.0
    combined = _noisy_or(list(category_scores.values()))
    overall = max(strongest, combined)

    if overall >= block_threshold:
        decision = Decision.BLOCK
    elif overall >= flag_threshold:
        decision = Decision.FLAG
    else:
        decision = Decision.ALLOW

    return ScoreResult(
        decision=decision,
        overall=overall,
        category_scores=category_scores,
        toxic_probability=toxic_probability,
        flag_threshold=flag_threshold,
        block_threshold=block_threshold,
    )
