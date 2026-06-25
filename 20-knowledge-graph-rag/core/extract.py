"""Entity + relation extractor, from scratch.

No spaCy, no NLTK, no model downloads. Entities are surfaced with three signals
that reinforce each other:

1. **Capitalization / proper-noun patterns** — runs of capitalized tokens
   (optionally joined by lowercase connectors like "of"/"and") that are not at a
   sentence start *only* are treated as proper nouns. We also keep a
   sentence-initial capitalized run when it survives a stop-word filter.
2. **A small gazetteer** — a hand-seeded set of known org/role/relation cue
   words that boosts precision and lets multi-word organization names with
   suffixes ("Inc", "Corp", "Ltd") merge greedily.
3. **Regex** — money, years, and acronym shapes.

Relations come from two sources:

* **Pattern rules** — verb templates like ``X acquired Y``, ``X founded Y``,
  ``X is the CEO of Y`` produce directed ``(head, relation, tail)`` triples.
* **Co-occurrence** — any two distinct entities inside the same sentence window
  are linked with a ``co_occurs`` relation whose weight is the running count of
  how often that pair appears together. This captures associations the verb
  templates miss.

Everything returns plain dataclasses / tuples so downstream code (the graph and
retriever) stays decoupled from how extraction happened.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

# A triple is (head_entity, relation_label, tail_entity).
Triple = Tuple[str, str, str]

# Tokens that should never start a standalone entity even when capitalized
# (sentence-initial function words, etc.).
_STOPWORDS: Set[str] = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "their",
    "his", "her", "our", "they", "he", "she", "we", "in", "on", "at", "of",
    "for", "to", "and", "but", "or", "with", "by", "from", "as", "is", "was",
    "were", "are", "be", "been", "after", "before", "when", "while", "who",
    "what", "which", "where", "how", "why", "also", "then", "unrelated",
}

# Lowercase connectors permitted *inside* a multi-word proper noun
# ("Bank of America", "Procter and Gamble").
_CONNECTORS: Set[str] = {"of", "and", "for", "the", "de", "&"}

# Org-name suffixes that greedily extend an entity span.
_ORG_SUFFIXES: Set[str] = {
    "inc", "inc.", "corp", "corp.", "ltd", "ltd.", "llc", "co", "co.",
    "company", "group", "holdings", "labs", "technologies", "systems",
    "ventures", "partners", "university", "institute",
}

# Seed gazetteer: role words and a couple of known entity hints. Kept tiny on
# purpose — the point is the *mechanism*, not coverage.
_GAZETTEER: Set[str] = {
    "CEO", "CTO", "CFO", "COO", "President", "Founder", "Chairman", "Director",
}

# Relation cue verbs mapped to a canonical relation label.
_RELATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bacquired\b", re.I), "acquired"),
    (re.compile(r"\bbought\b", re.I), "acquired"),
    (re.compile(r"\bfounded\b", re.I), "founded"),
    (re.compile(r"\bestablished\b", re.I), "founded"),
    (re.compile(r"\bmerged with\b", re.I), "merged_with"),
    (re.compile(r"\bpartnered with\b", re.I), "partnered_with"),
    (re.compile(r"\binvested in\b", re.I), "invested_in"),
    (re.compile(r"\bowns\b", re.I), "owns"),
    (re.compile(r"\bis a subsidiary of\b", re.I), "subsidiary_of"),
    (re.compile(r"\bis based in\b", re.I), "based_in"),
    (re.compile(r"\bheadquartered in\b", re.I), "based_in"),
]

# "X is the CEO of Y" style role relations.
_ROLE_RE = re.compile(
    r"(?P<head>[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*)*)\s+is\s+the\s+"
    r"(?P<role>CEO|CTO|CFO|COO|President|Founder|Chairman|Director)\s+of\s+"
    r"(?P<tail>[A-Z][\w.&'-]*(?:\s+(?:of|and|the)\s+)?(?:[A-Z][\w.&'-]*)*)",
    re.I,
)

_MONEY_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)?\s*dollars?\b", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[A-Za-z][\w.&'-]*")


@dataclass
class Extraction:
    """Result bundle from :func:`extract`.

    Attributes:
        entities: Sorted unique surface forms of detected entities.
        triples: Directed ``(head, relation, tail)`` triples from pattern rules.
        cooccurrence: Map of frozenset-pair -> count of co-occurring sentences.
    """

    entities: List[str] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    cooccurrence: Dict[Tuple[str, str], int] = field(default_factory=dict)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation (whitespace-aware)."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _is_capitalized(token: str) -> bool:
    return bool(token) and token[0].isupper()


def detect_entities(sentence: str) -> List[str]:
    """Detect entity surface forms in a single sentence.

    Greedily merges consecutive capitalized tokens, allowing a small set of
    lowercase connectors inside a span and absorbing org suffixes. Sentence-
    initial spans are kept only when the first token is not a stop word, so a
    leading "The" or "After" does not pollute the entity set.
    """
    tokens = sentence.split()
    entities: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        raw = tokens[i]
        word = raw.strip(",.;:()\"'")
        if not _is_capitalized(word):
            i += 1
            continue
        # Skip a sentence-initial capitalized stop word ("The", "After").
        if i == 0 and word.lower() in _STOPWORDS:
            i += 1
            continue
        # A gazetteer cue word ("CEO") is emitted on its own and never extends
        # a span, so it can't swallow the following org name ("CEO of Acme" -X-).
        if word in _GAZETTEER:
            i += 1
            continue
        span = [word]
        j = i + 1
        while j < n:
            nxt_raw = tokens[j]
            nxt = nxt_raw.strip(",.;:()\"'")
            low = nxt.lower()
            if _is_capitalized(nxt):
                span.append(nxt)
                j += 1
                continue
            # connector only if a capitalized token follows it
            if low in _CONNECTORS and j + 1 < n and _is_capitalized(
                tokens[j + 1].strip(",.;:()\"'")
            ):
                span.append(nxt)
                j += 1
                continue
            if low in _ORG_SUFFIXES:
                span.append(nxt)
                j += 1
            break
        surface = " ".join(span).strip(",.;:()\"'")
        # drop trailing connectors that slipped in
        while surface.split() and surface.split()[-1].lower() in _CONNECTORS:
            surface = surface.rsplit(" ", 1)[0]
        if surface and surface.lower() not in _STOPWORDS:
            entities.append(surface)
        i = max(j, i + 1)

    # gazetteer role words add themselves as entities too
    for cue in _GAZETTEER:
        if re.search(rf"\b{re.escape(cue)}\b", sentence):
            entities.append(cue)
    return entities


def _match_first_entity_after(text: str, entities: Sequence[str]) -> str:
    """Return the first entity surface form appearing in ``text``, else ''."""
    best, best_pos = "", len(text) + 1
    low = text.lower()
    for ent in entities:
        pos = low.find(ent.lower())
        if 0 <= pos < best_pos:
            best, best_pos = ent, pos
    return best


def extract_relations(sentence: str, entities: Sequence[str]) -> List[Triple]:
    """Extract directed ``(head, relation, tail)`` triples from one sentence.

    Uses verb-template patterns ("X acquired Y", "X founded Y") and an explicit
    role pattern ("X is the CEO of Y"). The head is the nearest entity before
    the cue verb; the tail is the nearest entity after it.
    """
    triples: List[Triple] = []

    # role pattern first (most specific)
    for m in _ROLE_RE.finditer(sentence):
        head = m.group("head").strip()
        role = m.group("role")
        tail = m.group("tail").strip().rstrip(",.;:")
        if head and tail:
            triples.append((head, f"is_{role.lower()}_of", tail))

    # verb templates: split around the cue, take nearest entity each side
    for pat, label in _RELATION_PATTERNS:
        m = pat.search(sentence)
        if not m:
            continue
        before, after = sentence[: m.start()], sentence[m.end():]
        head = _nearest_entity(before, entities, side="end")
        tail = _match_first_entity_after(after, entities)
        if head and tail and head != tail:
            triples.append((head, label, tail))
    return triples


def _nearest_entity(text: str, entities: Sequence[str], side: str = "end") -> str:
    """Return the entity surface form closest to the ``end`` of ``text``."""
    low = text.lower()
    best, best_pos = "", -1
    for ent in entities:
        pos = low.rfind(ent.lower())
        if pos > best_pos:
            best, best_pos = ent, pos
    return best


def extract(text: str) -> Extraction:
    """Run full extraction over a document.

    Returns entities, directed relation triples, and a co-occurrence count map.
    Co-occurrence pairs are stored as a sorted 2-tuple key so ``(A, B)`` and
    ``(B, A)`` accumulate into the same count.
    """
    all_entities: Set[str] = set()
    triples: List[Triple] = []
    cooc: Dict[Tuple[str, str], int] = defaultdict(int)

    for sent in split_sentences(text):
        ents = detect_entities(sent)
        uniq = list(dict.fromkeys(ents))  # preserve order, drop dups
        all_entities.update(uniq)
        triples.extend(extract_relations(sent, uniq))

        # co-occurrence within the sentence window
        for a_idx in range(len(uniq)):
            for b_idx in range(a_idx + 1, len(uniq)):
                a, b = uniq[a_idx], uniq[b_idx]
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                cooc[key] += 1

    return Extraction(
        entities=sorted(all_entities),
        triples=triples,
        cooccurrence=dict(cooc),
    )
