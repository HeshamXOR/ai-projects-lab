"""Inverted index with positional postings — from scratch.

An inverted index maps each term to the list of (document, positions) where it
occurs. It's the data structure under every search engine: instead of scanning
documents for a query, you intersect short postings lists. Positions enable
phrase queries ("exact sequence of words").

No Lucene/Whoosh — just dicts and lists, so the mechanics are visible.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class InvertedIndex:
    def __init__(self):
        # term -> {doc_id -> [positions]}
        self.postings: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.doc_len: Dict[int, int] = {}
        self.n_docs = 0

    def add(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        self.doc_len[doc_id] = len(tokens)
        for pos, term in enumerate(tokens):
            self.postings[term][doc_id].append(pos)
        self.n_docs = max(self.n_docs, doc_id + 1)

    def docs_with_term(self, term: str) -> Set[int]:
        return set(self.postings.get(term.lower(), {}).keys())

    def boolean_and(self, query: str) -> Set[int]:
        """Documents containing ALL query terms (postings-list intersection)."""
        terms = tokenize(query)
        if not terms:
            return set()
        result = self.docs_with_term(terms[0])
        for t in terms[1:]:
            result &= self.docs_with_term(t)
        return result

    def phrase_search(self, phrase: str) -> Set[int]:
        """Documents containing the terms as an adjacent sequence.

        Uses positions: a doc matches if there's a position p where every term
        i of the phrase appears at p+i.
        """
        terms = tokenize(phrase)
        if not terms:
            return set()
        candidates = self.boolean_and(phrase)
        matches = set()
        for doc in candidates:
            first_positions = self.postings[terms[0]][doc]
            for start in first_positions:
                if all(
                    (start + i) in self.postings[terms[i]].get(doc, [])
                    for i in range(1, len(terms))
                ):
                    matches.add(doc)
                    break
        return matches
