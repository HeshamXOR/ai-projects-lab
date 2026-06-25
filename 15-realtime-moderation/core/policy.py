"""Policy engine: evaluates text against a ruleset and collects hits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .rules import Category, Rule, RuleType, Span, default_ruleset


@dataclass
class RuleHit:
    """A single rule firing on a span of the input text.

    Attributes:
        rule_id: Identifier of the rule that fired.
        category: The category the rule belongs to.
        rule_type: Matcher family of the rule.
        severity: The rule's severity weight.
        start: Start index of the matched span.
        end: End index of the matched span.
        text: The matched substring.
        description: Human-readable rule description.
    """

    rule_id: str
    category: Category
    rule_type: RuleType
    severity: float
    start: int
    end: int
    text: str
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "category": self.category.value,
            "rule_type": self.rule_type.value,
            "severity": self.severity,
            "span": [self.start, self.end],
            "text": self.text,
            "description": self.description,
        }


class PolicyEngine:
    """Applies an ordered ruleset to text and gathers :class:`RuleHit` objects."""

    def __init__(self, rules: Sequence[Rule] | None = None) -> None:
        """Build an engine from ``rules`` (defaults to the bundled ruleset)."""
        self.rules: List[Rule] = list(rules) if rules is not None else default_ruleset()
        if not self.rules:
            raise ValueError("PolicyEngine requires at least one rule")
        # Sanity: rule ids must be unique so explanations are unambiguous.
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate rule ids: {sorted(dupes)}")

    def evaluate(self, text: str) -> List[RuleHit]:
        """Return every :class:`RuleHit` produced across all rules.

        Hits are returned in document order (by start index, then rule id) so
        that explanations are stable and readable.
        """
        hits: List[RuleHit] = []
        for rule in self.rules:
            for start, end, matched in rule.apply(text):
                hits.append(
                    RuleHit(
                        rule_id=rule.id,
                        category=rule.category,
                        rule_type=rule.rule_type,
                        severity=rule.severity,
                        start=start,
                        end=end,
                        text=matched,
                        description=rule.description,
                    )
                )
        hits.sort(key=lambda h: (h.start, h.rule_id))
        return hits

    def hits_by_category(self, text: str) -> Dict[Category, List[RuleHit]]:
        """Group :meth:`evaluate` results by category."""
        grouped: Dict[Category, List[RuleHit]] = {c: [] for c in Category}
        for hit in self.evaluate(text):
            grouped[hit.category].append(hit)
        return grouped
