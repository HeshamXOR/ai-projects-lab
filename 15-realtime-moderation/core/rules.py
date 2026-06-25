"""Rule DSL for the policy engine.

Rules are expressed as plain data (``Rule`` dataclasses, easily serializable
to/from dicts or YAML-like structures). Each rule carries a moderation
category and a severity weight, plus a typed matcher. Supported matcher types:

* ``KEYWORD``   -- whole-word match against any phrase in a list.
* ``REGEX``     -- arbitrary compiled regular expression.
* ``PII``       -- built-in detectors: email, phone, credit card (validated
                   with a from-scratch Luhn checksum), and US SSN.
* ``HEURISTIC`` -- spam / abuse heuristics: excessive caps, repeated
                   characters, repeated tokens, URL flooding.

A ``Rule`` produces zero or more ``(start, end, matched_text)`` spans when
applied; the policy engine turns those into ``RuleHit`` objects. This module
is standard-library only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple


class Category(str, Enum):
    """Moderation categories the engine can flag."""

    TOXICITY = "toxicity"
    PII = "pii"
    SPAM = "spam"
    SELF_HARM = "self_harm"


class RuleType(str, Enum):
    """The matcher family a rule belongs to."""

    KEYWORD = "keyword"
    REGEX = "regex"
    PII = "pii"
    HEURISTIC = "heuristic"


# A span is (start_index, end_index, matched_substring).
Span = Tuple[int, int, str]


# --------------------------------------------------------------------------- #
# From-scratch Luhn checksum for credit-card validation.
# --------------------------------------------------------------------------- #
def luhn_check(digits: str) -> bool:
    """Return ``True`` if ``digits`` passes the Luhn (mod-10) checksum.

    The Luhn algorithm, implemented by hand: walk the digits right-to-left,
    double every second digit, subtract 9 from any result above 9, sum the
    lot, and require the total to be divisible by 10.

    Args:
        digits: A string that may contain non-digit separators; they are
            stripped. Must contain at least 12 digits to be plausible.
    """
    only = [int(c) for c in digits if c.isdigit()]
    if len(only) < 12:
        return False
    total = 0
    # Index from the right: position 0 is the rightmost (check) digit.
    for pos, value in enumerate(reversed(only)):
        if pos % 2 == 1:  # every second digit from the right
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# --------------------------------------------------------------------------- #
# PII detector regexes (broad capture, then validated where applicable).
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
# Candidate card: 13-19 digits possibly separated by spaces or dashes.
_CARD_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _detect_email(text: str) -> List[Span]:
    return [(m.start(), m.end(), m.group()) for m in _EMAIL_RE.finditer(text)]


def _detect_phone(text: str) -> List[Span]:
    return [(m.start(), m.end(), m.group()) for m in _PHONE_RE.finditer(text)]


def _detect_ssn(text: str) -> List[Span]:
    return [(m.start(), m.end(), m.group()) for m in _SSN_RE.finditer(text)]


def _detect_credit_card(text: str) -> List[Span]:
    """Find card-like digit runs and keep only those passing Luhn."""
    spans: List[Span] = []
    for m in _CARD_CANDIDATE_RE.finditer(text):
        candidate = m.group()
        if luhn_check(candidate):
            spans.append((m.start(), m.end(), candidate))
    return spans


_PII_DETECTORS: dict[str, Callable[[str], List[Span]]] = {
    "email": _detect_email,
    "phone": _detect_phone,
    "ssn": _detect_ssn,
    "credit_card": _detect_credit_card,
}


# --------------------------------------------------------------------------- #
# Heuristic detectors (spam / abuse signals).
# --------------------------------------------------------------------------- #
def _heuristic_excessive_caps(text: str, min_len: int = 8, ratio: float = 0.7) -> List[Span]:
    """Flag the whole text when uppercase letters dominate."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_len:
        return []
    upper = sum(1 for c in letters if c.isupper())
    if upper / len(letters) >= ratio:
        return [(0, len(text), text)]
    return []


def _heuristic_repeated_chars(text: str, run: int = 5) -> List[Span]:
    """Flag runs of a single repeated character (``soooooo``, ``!!!!!!``)."""
    spans: List[Span] = []
    pattern = re.compile(r"(.)\1{" + str(run - 1) + r",}")
    for m in pattern.finditer(text):
        spans.append((m.start(), m.end(), m.group()))
    return spans


def _heuristic_repeated_tokens(text: str, repeats: int = 4) -> List[Span]:
    """Flag the same word repeated many times in a row."""
    spans: List[Span] = []
    pattern = re.compile(r"\b(\w+)(?:\s+\1\b){" + str(repeats - 1) + r",}", re.IGNORECASE)
    for m in pattern.finditer(text):
        spans.append((m.start(), m.end(), m.group()))
    return spans


def _heuristic_url_flood(text: str, threshold: int = 3) -> List[Span]:
    """Flag the text when it contains many URLs (link spam)."""
    urls = list(_URL_RE.finditer(text))
    if len(urls) >= threshold:
        return [(u.start(), u.end(), u.group()) for u in urls]
    return []


_HEURISTICS: dict[str, Callable[[str], List[Span]]] = {
    "excessive_caps": _heuristic_excessive_caps,
    "repeated_chars": _heuristic_repeated_chars,
    "repeated_tokens": _heuristic_repeated_tokens,
    "url_flood": _heuristic_url_flood,
}


@dataclass
class Rule:
    """A single moderation rule.

    Attributes:
        id: Stable identifier (used in explanations).
        category: The moderation category this rule contributes to.
        rule_type: Which matcher family to use.
        severity: Severity weight in ``[0, 1]`` added per hit during scoring.
        description: Human-readable summary for explanations.
        phrases: Phrase list for ``KEYWORD`` rules.
        pattern: Regex source for ``REGEX`` rules.
        detector: Detector key for ``PII`` / ``HEURISTIC`` rules.
        flags: Optional ``re`` flags for ``REGEX`` / ``KEYWORD``.
    """

    id: str
    category: Category
    rule_type: RuleType
    severity: float
    description: str = ""
    phrases: List[str] = field(default_factory=list)
    pattern: Optional[str] = None
    detector: Optional[str] = None
    flags: int = re.IGNORECASE

    _compiled: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if self.rule_type == RuleType.KEYWORD:
            if not self.phrases:
                raise ValueError(f"keyword rule {self.id!r} needs phrases")
            # Build one alternation regex with word boundaries; escape phrases.
            alternation = "|".join(re.escape(p) for p in self.phrases)
            self._compiled = re.compile(rf"\b(?:{alternation})\b", self.flags)
        elif self.rule_type == RuleType.REGEX:
            if not self.pattern:
                raise ValueError(f"regex rule {self.id!r} needs a pattern")
            self._compiled = re.compile(self.pattern, self.flags)
        elif self.rule_type == RuleType.PII:
            if self.detector not in _PII_DETECTORS:
                raise ValueError(f"unknown PII detector {self.detector!r}")
        elif self.rule_type == RuleType.HEURISTIC:
            if self.detector not in _HEURISTICS:
                raise ValueError(f"unknown heuristic {self.detector!r}")
        else:  # pragma: no cover - guarded by enum
            raise ValueError(f"unsupported rule_type {self.rule_type!r}")

    def apply(self, text: str) -> List[Span]:
        """Return all spans in ``text`` matched by this rule."""
        if self.rule_type in (RuleType.KEYWORD, RuleType.REGEX):
            assert self._compiled is not None
            return [
                (m.start(), m.end(), m.group())
                for m in self._compiled.finditer(text)
            ]
        if self.rule_type == RuleType.PII:
            return _PII_DETECTORS[self.detector](text)
        if self.rule_type == RuleType.HEURISTIC:
            return _HEURISTICS[self.detector](text)
        return []  # pragma: no cover


def default_ruleset() -> List[Rule]:
    """Return the bundled default ruleset covering all four categories."""
    return [
        # ---- Toxicity ----------------------------------------------------- #
        Rule(
            id="tox.insults",
            category=Category.TOXICITY,
            rule_type=RuleType.KEYWORD,
            severity=0.5,
            description="Common insults and slurs",
            phrases=[
                "idiot", "moron", "stupid", "dumb", "loser", "pathetic",
                "worthless", "garbage", "trash", "scum", "jerk", "fool",
                "disgusting", "vile", "brainless", "coward",
            ],
        ),
        Rule(
            id="tox.hate",
            category=Category.TOXICITY,
            rule_type=RuleType.KEYWORD,
            severity=0.7,
            description="Expressions of hatred or contempt",
            phrases=["i hate you", "i despise you", "you make me sick",
                     "nobody likes you", "everyone hates you"],
        ),
        Rule(
            id="tox.threat",
            category=Category.TOXICITY,
            rule_type=RuleType.REGEX,
            severity=0.9,
            description="Threats of violence",
            pattern=r"\bi\s+will\s+(?:hurt|kill|destroy|end|find)\s+you\b",
        ),
        # ---- Self-harm ---------------------------------------------------- #
        Rule(
            id="sh.directives",
            category=Category.SELF_HARM,
            rule_type=RuleType.KEYWORD,
            severity=0.95,
            description="Self-harm or suicide directives",
            phrases=["kill yourself", "kys", "end your life",
                     "you should die", "hurt yourself"],
        ),
        Rule(
            id="sh.ideation",
            category=Category.SELF_HARM,
            rule_type=RuleType.REGEX,
            severity=0.8,
            description="First-person self-harm ideation",
            pattern=r"\bi\s+(?:want|need)\s+to\s+(?:die|kill myself|end it)\b",
        ),
        # ---- PII ---------------------------------------------------------- #
        Rule(id="pii.email", category=Category.PII, rule_type=RuleType.PII,
             severity=0.4, description="Email address", detector="email"),
        Rule(id="pii.phone", category=Category.PII, rule_type=RuleType.PII,
             severity=0.4, description="Phone number", detector="phone"),
        Rule(id="pii.credit_card", category=Category.PII, rule_type=RuleType.PII,
             severity=0.9, description="Credit card number (Luhn-valid)",
             detector="credit_card"),
        Rule(id="pii.ssn", category=Category.PII, rule_type=RuleType.PII,
             severity=0.9, description="US Social Security Number",
             detector="ssn"),
        # ---- Spam --------------------------------------------------------- #
        Rule(id="spam.caps", category=Category.SPAM, rule_type=RuleType.HEURISTIC,
             severity=0.3, description="Excessive capitalization",
             detector="excessive_caps"),
        Rule(id="spam.repeated_chars", category=Category.SPAM,
             rule_type=RuleType.HEURISTIC, severity=0.25,
             description="Repeated characters", detector="repeated_chars"),
        Rule(id="spam.repeated_tokens", category=Category.SPAM,
             rule_type=RuleType.HEURISTIC, severity=0.35,
             description="Repeated tokens", detector="repeated_tokens"),
        Rule(id="spam.url_flood", category=Category.SPAM,
             rule_type=RuleType.HEURISTIC, severity=0.5,
             description="URL flooding / link spam", detector="url_flood"),
        Rule(id="spam.promo", category=Category.SPAM, rule_type=RuleType.KEYWORD,
             severity=0.4, description="Promotional spam phrases",
             phrases=["buy now", "click here", "free money", "act now",
                      "limited offer", "you have won", "claim your prize"]),
    ]
