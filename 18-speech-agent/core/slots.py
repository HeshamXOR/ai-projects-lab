"""Slot filling: extract entities from an utterance for a given intent.

Each intent declares which slots it *requires* and *optionally* accepts. Slots
are extracted with regex + keyword patterns (dates, times, numbers, cities,
named slots). ``fill_slots`` returns the slots it could fill plus the required
ones still missing — the signal the dialog FSM uses to decide whether to ask a
follow-up question.

From scratch: no NER model, just transparent rule-based extractors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# A small, deterministic gazetteer of cities. Real systems would use a larger
# lexicon or an NER model; this keeps extraction testable and offline.
_CITIES = {
    "london", "paris", "berlin", "madrid", "rome", "tokyo", "cairo", "dubai",
    "new york", "san francisco", "los angeles", "boston", "chicago", "seattle",
    "amsterdam", "lisbon", "vienna", "athens", "oslo", "dublin", "doha",
}

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

_RELATIVE_DAYS = {"today", "tomorrow", "tonight", "yesterday"}
_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


def extract_number(text: str) -> Optional[int]:
    """Return the first integer in ``text`` (digits or number words), else None."""
    digit = re.search(r"\b\d+\b", text)
    if digit:
        return int(digit.group())
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text.lower()):
            return value
    return None


def extract_date(text: str) -> Optional[str]:
    """Return a normalized date-ish phrase if one is present, else None."""
    low = text.lower()
    # ISO date e.g. 2026-06-25
    iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if iso:
        return iso.group()
    # numeric date e.g. 12/05 or 12/05/2026
    numeric = re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text)
    if numeric:
        return numeric.group()
    # "june 25" / "25 june" / "june 25th"
    month_first = re.search(
        rf"\b({'|'.join(_MONTHS)})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", low
    )
    if month_first:
        return month_first.group()
    day_first = re.search(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+({'|'.join(_MONTHS)})\b", low
    )
    if day_first:
        return day_first.group()
    for token in list(_RELATIVE_DAYS) + list(_WEEKDAYS):
        if re.search(rf"\b{token}\b", low):
            return token
    return None


def extract_time(text: str) -> Optional[str]:
    """Return a clock time like ``3pm`` or ``14:30`` if present, else None."""
    clock = re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", text.lower())
    if clock:
        return clock.group().strip()
    ampm = re.search(r"\b\d{1,2}\s*(?:am|pm)\b", text.lower())
    if ampm:
        return ampm.group().strip()
    return None


def extract_city(text: str) -> Optional[str]:
    """Return the first known city mentioned in ``text``, else None."""
    low = text.lower()
    # check multi-word cities first so "new york" wins over "new"
    for city in sorted(_CITIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", low):
            return city
    return None


# Registry of named extractors so intents can reference them by slot name.
EXTRACTORS: Dict[str, Callable[[str], Optional[object]]] = {
    "date": extract_date,
    "time": extract_time,
    "number": extract_number,
    "city": extract_city,
    "origin": extract_city,
    "destination": extract_city,
}


@dataclass
class IntentSchema:
    """Declares the slots an intent needs and which extractor fills each."""

    name: str
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    # per-slot extractor override; defaults to EXTRACTORS[slot]
    extractors: Dict[str, Callable[[str], Optional[object]]] = field(default_factory=dict)

    def extractor_for(self, slot: str) -> Callable[[str], Optional[object]]:
        """Return the extractor function for ``slot``."""
        if slot in self.extractors:
            return self.extractors[slot]
        return EXTRACTORS.get(slot, lambda _t: None)


# Default schemas for the demo intents.
DEFAULT_SCHEMAS: Dict[str, IntentSchema] = {
    "greet": IntentSchema("greet"),
    "goodbye": IntentSchema("goodbye"),
    "book_flight": IntentSchema(
        "book_flight",
        required=["destination", "date"],
        optional=["origin"],
    ),
    "check_weather": IntentSchema(
        "check_weather",
        required=["city"],
        optional=["date"],
    ),
    "set_alarm": IntentSchema(
        "set_alarm",
        required=["time"],
        optional=["date"],
    ),
}


@dataclass
class SlotResult:
    """Outcome of slot extraction for one utterance under one intent."""

    intent: str
    slots: Dict[str, object]
    missing: List[str]


def fill_slots(
    intent: str,
    text: str,
    schemas: Optional[Dict[str, IntentSchema]] = None,
    known: Optional[Dict[str, object]] = None,
) -> SlotResult:
    """Extract slots for ``intent`` from ``text``.

    ``known`` carries slots already filled in earlier turns (so a follow-up that
    only supplies the missing value still completes the frame). Returns the
    merged slot dict and the list of required slots still empty.
    """
    schemas = schemas or DEFAULT_SCHEMAS
    schema = schemas.get(intent, IntentSchema(intent))
    filled: Dict[str, object] = dict(known or {})

    for slot in schema.required + schema.optional:
        if slot in filled and filled[slot] is not None:
            continue  # already known from context
        value = schema.extractor_for(slot)(text)
        if value is not None:
            filled[slot] = value

    missing = [s for s in schema.required if filled.get(s) is None]
    return SlotResult(intent=intent, slots=filled, missing=missing)
