"""Structured explanations for moderation decisions.

Given the rule hits, the classifier probability, and the score result, build a
machine-readable explanation describing *why* a verdict was reached: which
rules fired, on which spans, and how much the classifier contributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .policy import RuleHit
from .scoring import ScoreResult


@dataclass
class Explanation:
    """A structured, serializable rationale for a decision.

    Attributes:
        summary: One-line human-readable summary.
        decision: The decision string (``allow`` / ``flag`` / ``block``).
        triggered_rules: Per-rule detail (id, category, span, matched text).
        category_breakdown: Per-category score plus contributing rule ids.
        classifier: The classifier's contribution to the toxicity signal.
    """

    summary: str
    decision: str
    triggered_rules: List[Dict[str, object]]
    category_breakdown: Dict[str, Dict[str, object]]
    classifier: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "summary": self.summary,
            "decision": self.decision,
            "triggered_rules": self.triggered_rules,
            "category_breakdown": self.category_breakdown,
            "classifier": self.classifier,
        }


def build_explanation(
    text: str,
    hits: Sequence[RuleHit],
    score: ScoreResult,
) -> Explanation:
    """Assemble an :class:`Explanation` from the pipeline's intermediate state.

    Args:
        text: The original input (used to render readable spans).
        hits: Rule hits from the policy engine.
        score: The :class:`ScoreResult` produced by scoring.
    """
    triggered: List[Dict[str, object]] = []
    rules_by_cat: Dict[str, List[str]] = {}
    for hit in hits:
        triggered.append(
            {
                "rule_id": hit.rule_id,
                "category": hit.category.value,
                "rule_type": hit.rule_type.value,
                "severity": hit.severity,
                "span": [hit.start, hit.end],
                "matched_text": hit.text,
                "description": hit.description,
            }
        )
        rules_by_cat.setdefault(hit.category.value, []).append(hit.rule_id)

    category_breakdown: Dict[str, Dict[str, object]] = {}
    for cat, cat_score in score.category_scores.items():
        category_breakdown[cat] = {
            "score": round(cat_score, 4),
            "rules": rules_by_cat.get(cat, []),
        }

    classifier_info = {
        "toxic_probability": round(score.toxic_probability, 4),
        "contributed_to": "toxicity",
        "note": (
            "Naive Bayes P(toxic) is folded into the toxicity category score "
            "via the configured classifier weight."
        ),
    }

    # Build a readable summary.
    if not hits and score.toxic_probability < 0.5:
        summary = "No rules fired and the classifier judged the text clean."
    else:
        fired_cats = sorted({h.category.value for h in hits})
        parts: List[str] = []
        if fired_cats:
            parts.append(f"rules fired for: {', '.join(fired_cats)}")
        if score.toxic_probability >= 0.5:
            parts.append(
                f"classifier P(toxic)={score.toxic_probability:.2f}"
            )
        summary = (
            f"Decision={score.decision.value.upper()} "
            f"(overall={score.overall:.2f}); " + "; ".join(parts) + "."
        )

    return Explanation(
        summary=summary,
        decision=score.decision.value,
        triggered_rules=triggered,
        category_breakdown=category_breakdown,
        classifier=classifier_info,
    )
