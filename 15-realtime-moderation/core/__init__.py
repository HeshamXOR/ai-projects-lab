"""Real-time text moderation core.

A from-scratch moderation engine: tokenizer, multinomial Naive Bayes
classifier, a rule DSL + policy engine, severity scoring, explanations,
and an orchestrating pipeline. The entire core is standard-library only.
"""

from .tokenizer import Tokenizer
from .naive_bayes import MultinomialNaiveBayes
from .rules import Rule, default_ruleset, Category, RuleType
from .policy import PolicyEngine, RuleHit
from .scoring import score_text, Decision, ScoreResult
from .explain import build_explanation, Explanation
from .pipeline import ModerationPipeline, ModerationResult

__all__ = [
    "Tokenizer",
    "MultinomialNaiveBayes",
    "Rule",
    "default_ruleset",
    "Category",
    "RuleType",
    "PolicyEngine",
    "RuleHit",
    "score_text",
    "Decision",
    "ScoreResult",
    "build_explanation",
    "Explanation",
    "ModerationPipeline",
    "ModerationResult",
]

__version__ = "1.0.0"
