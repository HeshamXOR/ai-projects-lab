"""Citation verifier: check that each claim is grounded in a retrieved source.

A RAG answer is only trustworthy if its claims actually trace to the documents.
This verifier splits an answer into sentences and, for each, finds the
best-supporting source passage by lexical overlap (Jaccard over content words).
Sentences whose best support is below a threshold are flagged as **unsupported**
— the honest "this part might be hallucinated" signal.

From scratch (no NLI model needed); an embedding/NLI check is a drop-in upgrade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_STOP = set(
    "the a an and or of to in on for with is are was were be been being this that "
    "these those it its as at by from we you they i he she them his her their our".split()
)


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    score: float
    best_source: int   # index of the best-supporting passage, -1 if none


def verify_answer(answer: str, sources: List[str], threshold: float = 0.18) -> List[ClaimCheck]:
    """Check each sentence of `answer` against the `sources`."""
    source_words = [_content_words(s) for s in sources]
    checks: List[ClaimCheck] = []
    for sent in _split_sentences(answer):
        cw = _content_words(sent)
        if not cw:
            continue
        best_score, best_idx = 0.0, -1
        for i, sw in enumerate(source_words):
            if not sw:
                continue
            jac = len(cw & sw) / len(cw | sw)
            if jac > best_score:
                best_score, best_idx = jac, i
        checks.append(ClaimCheck(sent, best_score >= threshold, round(best_score, 3), best_idx))
    return checks


def groundedness(checks: List[ClaimCheck]) -> float:
    """Fraction of claims that are supported — a single trust score."""
    if not checks:
        return 0.0
    return sum(c.supported for c in checks) / len(checks)
