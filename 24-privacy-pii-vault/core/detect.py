"""From-scratch PII detection engine.

WHY this module exists
----------------------
This is the heart of the vault. It locates personally-identifiable
information in free text *without* any ML library (no spaCy / presidio). The
design combines two complementary strategies, because different PII classes
have fundamentally different structure:

  * **Structured PII** (email, phone, SSN, credit card, IP, dates) has a
    learnable surface shape, so we use carefully bounded regular expressions
    plus *validators* (e.g. the Luhn checksum for cards, octet-range checks
    for IPs). The validator step is what turns a loose pattern into a
    high-precision detector -- a 16-digit run that fails Luhn is rejected.

  * **Unstructured PII** (PERSON, ORG) is an open class with no fixed shape,
    so we lean on the gazetteer + context-cue approach implemented in
    ``gazetteer.py``: capitalized-token runs are scored by gazetteer
    membership and surrounding cues (honorifics, org suffixes, triggers).

Every hit is returned as an immutable :class:`Detection` with a type, a
character span ``[start, end)`` into the *original* string, the matched
value, and a calibrated confidence in ``[0, 1]``. Downstream code (the policy
engine) consumes spans, so overlapping/adjacent detections are resolved here:
higher-confidence and longer spans win, guaranteeing the policy engine can
apply transforms left-to-right without corrupting offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Pattern

from . import gazetteer


class PIIType(str, Enum):
    """Enumeration of PII categories the detector can emit.

    Inherits from ``str`` so values serialize cleanly to JSON and compare
    naturally with plain strings in tests and policy maps.
    """

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    DATE = "DATE"
    PERSON = "PERSON"
    ORG = "ORG"


@dataclass(frozen=True)
class Detection:
    """A single PII finding.

    Attributes
    ----------
    type:
        The :class:`PIIType` category.
    start, end:
        Character offsets into the original text; the matched substring is
        ``text[start:end]`` (half-open interval).
    value:
        The exact matched substring.
    confidence:
        Calibrated score in ``[0, 1]``. Regex+validator hits score high;
        gazetteer/context PERSON/ORG hits score by evidence strength.
    """

    type: PIIType
    start: int
    end: int
    value: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span [{self.start},{self.end})")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")

    @property
    def length(self) -> int:
        """Number of characters covered by the detection span."""
        return self.end - self.start

    def overlaps(self, other: "Detection") -> bool:
        """Return True if this detection's span intersects ``other``'s."""
        return self.start < other.end and other.start < self.end


# ---------------------------------------------------------------------------
# Luhn checksum -- implemented from scratch (no library).
# ---------------------------------------------------------------------------
def luhn_is_valid(number: str) -> bool:
    """Validate a credit-card-like number with the Luhn (mod-10) algorithm.

    The Luhn algorithm detects single-digit errors and most transpositions.
    Procedure, walking right-to-left:

      * every second digit (the 2nd, 4th, ... from the right) is doubled;
      * if a doubled value exceeds 9, subtract 9 (equivalent to summing its
        two decimal digits);
      * the total of all resulting digits must be a multiple of 10.

    Non-digit characters (spaces, dashes) are ignored so that grouped forms
    like ``4111-1111-1111-1111`` validate. Returns ``False`` for inputs with
    fewer than 12 digits, which rules out short numeric runs masquerading as
    cards.
    """
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    # Process from rightmost digit; double every second one.
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit
    return total % 10 == 0


def _valid_ipv4(candidate: str) -> bool:
    """Return True if ``candidate`` is a dotted-quad with octets 0-255."""
    parts = candidate.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        # Reject leading zeros like "01" which are ambiguous, except "0".
        if len(part) > 1 and part[0] == "0":
            return False
        if int(part) > 255:
            return False
    return True


# ---------------------------------------------------------------------------
# Compiled regular expressions for structured PII.
# ---------------------------------------------------------------------------
# Email: a pragmatic RFC-5322 subset. Local part allows common specials;
# domain requires at least one dot and a 2+ char TLD.
_EMAIL_RE: Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b"
)

# US/international phone: optional +country, optional area-code parens,
# separators of space/dot/dash. Requires enough digits to avoid matching
# bare years.
_PHONE_RE: Pattern[str] = re.compile(
    r"(?<![\w.])(?:\+?\d{1,3}[\s.\-]?)?"        # optional country code
    r"(?:\(\d{3}\)|\d{3})[\s.\-]?"               # area code, maybe parens
    r"\d{3}[\s.\-]?\d{4}"                         # local number
    r"(?![\w])"
)

# SSN: NNN-NN-NNNN. Excludes invalid area numbers 000, 666, 900-999 and
# group/serial 00 / 0000 per SSA rules -- raises precision.
_SSN_RE: Pattern[str] = re.compile(
    r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
)

# Credit card: 13-19 digit runs allowing space/dash grouping. Validity is
# confirmed by Luhn afterwards (regex alone is intentionally loose).
_CARD_RE: Pattern[str] = re.compile(
    r"(?<![\w\-])(?:\d[ \-]?){12,18}\d(?![\w\-])"
)

# IPv4: loose dotted-quad; octet ranges validated by ``_valid_ipv4``.
_IP_RE: Pattern[str] = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# Dates: ISO (YYYY-MM-DD), US (M/D/YYYY), and "Month DD, YYYY".
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE: Pattern[str] = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}"                                   # 2023-05-17
    r"|\d{1,2}/\d{1,2}/\d{2,4}"                            # 5/17/2023
    r"|(?:" + _MONTHS + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"  # May 17, 2023
    r")\b",
    re.IGNORECASE,
)

# A capitalized token: starts uppercase, rest letters/hyphen/apostrophe.
_CAP_TOKEN_RE: Pattern[str] = re.compile(r"[A-Z][a-zA-Z'\-]+\.?")
# Generic word tokenizer for context look-back.
_WORD_RE: Pattern[str] = re.compile(r"\S+")


def _detect_structured(text: str) -> List[Detection]:
    """Run all regex+validator passes over ``text`` and collect detections."""
    out: List[Detection] = []

    for m in _EMAIL_RE.finditer(text):
        out.append(Detection(PIIType.EMAIL, m.start(), m.end(), m.group(), 0.99))

    for m in _SSN_RE.finditer(text):
        out.append(Detection(PIIType.SSN, m.start(), m.end(), m.group(), 0.97))

    # Credit cards: regex is loose, so each candidate must pass Luhn.
    for m in _CARD_RE.finditer(text):
        candidate = m.group()
        if luhn_is_valid(candidate):
            out.append(
                Detection(PIIType.CREDIT_CARD, m.start(), m.end(), candidate, 0.98)
            )

    for m in _PHONE_RE.finditer(text):
        # Require at least 10 digits to avoid false hits on plain numbers.
        if sum(c.isdigit() for c in m.group()) >= 10:
            out.append(
                Detection(PIIType.PHONE, m.start(), m.end(), m.group(), 0.9)
            )

    for m in _IP_RE.finditer(text):
        if _valid_ipv4(m.group()):
            out.append(
                Detection(PIIType.IP_ADDRESS, m.start(), m.end(), m.group(), 0.95)
            )

    for m in _DATE_RE.finditer(text):
        out.append(Detection(PIIType.DATE, m.start(), m.end(), m.group(), 0.85))

    return out


def _detect_known_orgs(text: str) -> List[Detection]:
    """Match multi-word known organizations from the gazetteer directly."""
    out: List[Detection] = []
    lowered = text.lower()
    for org in gazetteer.KNOWN_ORGS:
        start = 0
        while True:
            idx = lowered.find(org, start)
            if idx == -1:
                break
            # Enforce word boundaries around the match.
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after = idx + len(org)
            after_ok = after >= len(text) or not text[after].isalnum()
            if before_ok and after_ok:
                out.append(
                    Detection(PIIType.ORG, idx, after, text[idx:after], 0.95)
                )
            start = idx + len(org)
    return out


@dataclass
class _Token:
    """A whitespace-delimited token with its character span."""

    text: str
    start: int
    end: int


def _tokenize(text: str) -> List[_Token]:
    """Split ``text`` into tokens, recording original character offsets."""
    return [_Token(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _detect_names_orgs(text: str) -> List[Detection]:
    """Detect PERSON and ORG entities via capitalized-run + context scoring.

    Algorithm
    ---------
    1. Tokenize, keeping offsets.
    2. Scan for maximal runs of capitalized tokens (a person/org candidate is
       usually a contiguous run like "John Smith" or "Acme Technologies").
    3. Score the run:
         * gazetteer hits on given-name / surname tokens -> PERSON evidence;
         * an org keyword token (Inc, LLC, Technologies) anywhere in or right
           after the run -> ORG;
         * a preceding honorific ("Dr.") -> PERSON even with novel tokens;
         * a preceding trigger ("works at", "named") nudges the type.
    4. Confidence is a function of how much evidence fired.
    """
    out: List[Detection] = []
    tokens = _tokenize(text)
    n = len(tokens)
    i = 0

    while i < n:
        tok = tokens[i]
        first_char = tok.text[0] if tok.text else ""
        if not first_char.isupper():
            i += 1
            continue

        # Greedily extend a run of capitalized tokens (allow internal org kw).
        j = i
        while j < n and tokens[j].text[:1].isupper():
            j += 1
        run = tokens[i:j]
        run_text = text[run[0].start : run[-1].end]

        # ---- Context look-back (previous 1-2 tokens) -------------------
        prev1 = tokens[i - 1].text if i - 1 >= 0 else ""
        prev2 = tokens[i - 2].text if i - 2 >= 0 else ""
        prev_honorific = gazetteer.is_honorific(prev1)
        prev_phrase = f"{prev2} {prev1}".lower().strip()
        person_trigger = (
            prev1.lower().strip(".,") in gazetteer.PERSON_TRIGGERS
            or prev_phrase in gazetteer.PERSON_TRIGGERS
        )
        org_trigger = (
            prev1.lower() in gazetteer.ORG_CONTEXT_TRIGGERS
            or prev_phrase in gazetteer.ORG_CONTEXT_TRIGGERS
        )

        # ---- Gazetteer evidence over the run ----------------------------
        given_hits = sum(1 for t in run if gazetteer.is_given_name(t.text))
        surname_hits = sum(1 for t in run if gazetteer.is_surname(t.text))
        org_kw_hits = sum(1 for t in run if gazetteer.is_org_token(t.text))
        # Also peek at the token *after* the run for a trailing "Inc/LLC".
        trailing_org = j < n and gazetteer.is_org_token(tokens[j].text)

        detection: Optional[Detection] = None

        if org_kw_hits or trailing_org:
            # Extend the span to include a trailing org keyword if present.
            end_off = tokens[j].end if trailing_org else run[-1].end
            value = text[run[0].start : end_off]
            conf = 0.9 if (org_kw_hits + int(trailing_org)) >= 1 else 0.7
            if org_trigger:
                conf = min(1.0, conf + 0.05)
            detection = Detection(PIIType.ORG, run[0].start, end_off, value, conf)
        elif prev_honorific:
            # Honorific guarantees a person even for unknown surnames.
            conf = 0.95 if (given_hits or surname_hits) else 0.85
            detection = Detection(
                PIIType.PERSON, run[0].start, run[-1].end, run_text, conf
            )
        elif given_hits or surname_hits:
            # Gazetteer-backed person. Two-token "Given Surname" is strongest.
            evidence = given_hits + surname_hits
            if given_hits and surname_hits:
                conf = 0.92
            elif evidence >= 1 and len(run) >= 2:
                conf = 0.8
            else:
                conf = 0.7
            if person_trigger:
                conf = min(1.0, conf + 0.05)
            detection = Detection(
                PIIType.PERSON, run[0].start, run[-1].end, run_text, conf
            )
        elif person_trigger and len(run) >= 1:
            # Trigger like "named" before a novel capitalized token.
            detection = Detection(
                PIIType.PERSON, run[0].start, run[-1].end, run_text, 0.6
            )

        if detection is not None:
            out.append(detection)

        i = j  # continue after the run

    return out


def _resolve_overlaps(detections: List[Detection]) -> List[Detection]:
    """Remove overlapping detections, preferring higher confidence then span.

    The policy engine applies transforms by character span, so overlaps would
    corrupt offsets. We sort by (confidence desc, length desc) and greedily
    keep a detection only if it does not overlap an already-kept one.
    """
    ordered = sorted(
        detections, key=lambda d: (d.confidence, d.length), reverse=True
    )
    kept: List[Detection] = []
    for det in ordered:
        if not any(det.overlaps(k) for k in kept):
            kept.append(det)
    # Return in document order for stable, readable output.
    kept.sort(key=lambda d: d.start)
    return kept


@dataclass
class Detector:
    """Configurable PII detector.

    Parameters
    ----------
    min_confidence:
        Detections scoring below this threshold are discarded. Lets callers
        trade recall for precision without touching the rules.
    enabled_types:
        Optional whitelist of :class:`PIIType` to emit; ``None`` emits all.
    """

    min_confidence: float = 0.5
    enabled_types: Optional[frozenset] = None

    def detect(self, text: str) -> List[Detection]:
        """Detect all PII in ``text`` and return resolved, sorted detections."""
        if not text:
            return []
        raw: List[Detection] = []
        raw.extend(_detect_structured(text))
        raw.extend(_detect_known_orgs(text))
        raw.extend(_detect_names_orgs(text))

        filtered = [d for d in raw if d.confidence >= self.min_confidence]
        if self.enabled_types is not None:
            filtered = [d for d in filtered if d.type in self.enabled_types]
        return _resolve_overlaps(filtered)


def detect(text: str, min_confidence: float = 0.5) -> List[Detection]:
    """Module-level convenience wrapper around :class:`Detector`."""
    return Detector(min_confidence=min_confidence).detect(text)
