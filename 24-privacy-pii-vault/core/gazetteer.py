"""From-scratch name/organization gazetteers and context-cue lexicons.

WHY this module exists
----------------------
A pure regex detector cannot recognize *names* and *organizations*, because
those are open classes with no fixed syntactic shape. Real NER systems learn
this from data; here we implement a small, explainable substitute:

  1. A curated gazetteer of common given names, surnames, and organization
     tokens. Membership in the gazetteer is one strong feature.
  2. Context cues -- honorifics ("Mr.", "Dr."), org suffixes ("Inc", "LLC"),
     and trigger phrases ("works at", "employed by") -- that boost or create a
     detection even when a token is *not* in the gazetteer (handles novel
     names like "Zyxwq" appearing after "Dr.").

Keeping these as plain frozensets makes the matching logic in ``detect.py``
trivial to reason about and unit-test, and keeps the whole detector dependency
free (no spaCy / presidio). The lists are intentionally small but cover the
labeled-sample vocabulary the tests exercise; in production you would load a
much larger gazetteer from disk.
"""

from __future__ import annotations

from typing import FrozenSet

# ---------------------------------------------------------------------------
# Given names (first names). Lowercased for case-insensitive membership tests.
# ---------------------------------------------------------------------------
GIVEN_NAMES: FrozenSet[str] = frozenset(
    {
        "james", "john", "robert", "michael", "william", "david", "richard",
        "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
        "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
        "susan", "jessica", "sarah", "karen", "nancy", "lisa", "margaret",
        "betty", "sandra", "ashley", "dorothy", "kimberly", "emily", "donna",
        "alice", "bob", "carol", "carlos", "maria", "jose", "ana", "luis",
        "wei", "li", "chen", "priya", "raj", "amit", "fatima", "omar",
        "olivia", "noah", "liam", "emma", "ava", "sophia", "grace", "henry",
        "alan", "alana", "alex", "alexandra", "frank", "george", "harry",
        "kevin", "laura", "nina", "oscar", "peter", "quinn", "rachel",
        "samuel", "tina", "victor", "wendy", "xavier", "yusuf", "zoe",
    }
)

# ---------------------------------------------------------------------------
# Surnames (family names).
# ---------------------------------------------------------------------------
SURNAMES: FrozenSet[str] = frozenset(
    {
        "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
        "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez",
        "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin",
        "lee", "perez", "thompson", "white", "harris", "sanchez", "clark",
        "ramirez", "lewis", "robinson", "walker", "young", "allen", "king",
        "wright", "scott", "torres", "nguyen", "hill", "flores", "green",
        "adams", "nelson", "baker", "hall", "rivera", "campbell", "mitchell",
        "carter", "roberts", "patel", "kim", "chen", "wang", "kumar", "singh",
        "okafor", "mwangi", "doe", "bauer", "novak", "rossi", "santos",
    }
)

# ---------------------------------------------------------------------------
# Organization keyword tokens. If a capitalized-token run contains one of
# these (or is immediately followed / preceded by one), it is an ORG.
# ---------------------------------------------------------------------------
ORG_TOKENS: FrozenSet[str] = frozenset(
    {
        "inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "co", "co.",
        "company", "corporation", "incorporated", "limited", "group",
        "holdings", "partners", "associates", "industries", "enterprises",
        "technologies", "systems", "solutions", "labs", "laboratories",
        "foundation", "institute", "university", "college", "bank", "trust",
        "capital", "ventures", "consulting", "services", "international",
        "global", "worldwide", "gmbh", "ag", "plc", "pty",
    }
)

# ---------------------------------------------------------------------------
# Well-known organization full names (multi-word) for direct matching.
# Stored lowercased; matched as contiguous spans.
# ---------------------------------------------------------------------------
KNOWN_ORGS: FrozenSet[str] = frozenset(
    {
        "acme corporation", "acme corp", "globex", "initech", "umbrella corp",
        "stark industries", "wayne enterprises", "openai", "anthropic",
        "google", "microsoft", "amazon", "apple", "meta", "netflix",
        "world health organization", "united nations", "red cross",
    }
)

# ---------------------------------------------------------------------------
# Honorific / title cues that immediately precede a personal name.
# ---------------------------------------------------------------------------
HONORIFICS: FrozenSet[str] = frozenset(
    {
        "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.",
        "prof", "prof.", "professor", "sir", "madam", "lord", "lady",
        "rev", "rev.", "fr", "fr.", "capt", "capt.", "sgt", "sgt.",
        "lt", "lt.", "col", "col.", "gen", "gen.", "hon", "hon.",
    }
)

# ---------------------------------------------------------------------------
# Trigger phrases (lowercased, may be multi-word) that signal a following
# PERSON or ORG. The detector looks back a few tokens for these.
# ---------------------------------------------------------------------------
PERSON_TRIGGERS: FrozenSet[str] = frozenset(
    {"named", "called", "name is", "i am", "this is", "signed", "regards",
     "sincerely", "contact", "attn", "patient", "customer", "client"}
)

ORG_CONTEXT_TRIGGERS: FrozenSet[str] = frozenset(
    {"works at", "employed by", "employee of", "ceo of", "founder of",
     "joined", "from", "at", "company", "firm", "organization"}
)


def is_given_name(token: str) -> bool:
    """Return True if ``token`` is a known given name (case-insensitive)."""
    return token.lower().strip(".,;:") in GIVEN_NAMES


def is_surname(token: str) -> bool:
    """Return True if ``token`` is a known surname (case-insensitive)."""
    return token.lower().strip(".,;:") in SURNAMES


def is_org_token(token: str) -> bool:
    """Return True if ``token`` is an organization keyword (e.g. 'Inc')."""
    return token.lower().strip(",;:") in ORG_TOKENS


def is_honorific(token: str) -> bool:
    """Return True if ``token`` is an honorific/title (e.g. 'Dr.')."""
    return token.lower() in HONORIFICS
