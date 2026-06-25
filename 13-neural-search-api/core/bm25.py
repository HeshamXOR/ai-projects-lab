"""A BM25 lexical index implemented from scratch.

BM25 (Best Matching 25) is the classic probabilistic ranking function for
keyword search. This module implements it end to end -- tokenization, term
frequency bookkeeping, inverse document frequency, document-length
normalization, and the final Okapi BM25 score -- with no external search
library.

For a query :math:`Q` containing terms :math:`q_1 \\dots q_n` and a document
:math:`D`, the score is::

    score(D, Q) = sum_i IDF(q_i) * ( f(q_i, D) * (k1 + 1) )
                  / ( f(q_i, D) + k1 * (1 - b + b * |D| / avgdl) )

where ``f(q_i, D)`` is the term frequency of ``q_i`` in ``D``, ``|D|`` is the
document length in tokens, ``avgdl`` is the average document length, and the
IDF uses the standard BM25 probabilistic form::

    IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )

with ``N`` the number of documents and ``n(q)`` the number of documents
containing ``q``. The ``+1`` inside the log keeps IDF non-negative even for
terms that appear in more than half of the corpus.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from .embed import tokenize


class BM25Index:
    """An incremental, in-memory BM25 index.

    The index keeps per-document term frequencies and a global postings map
    (term -> set of document ids) so that querying only touches documents
    that actually contain at least one query term.

    Args:
        k1: Term-frequency saturation parameter. Typical range 1.2-2.0.
        b: Document-length normalization strength in ``[0, 1]``.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)

        # doc_id -> {term: frequency}
        self._tf: Dict[str, Dict[str, int]] = {}
        # doc_id -> document length in tokens
        self._doc_len: Dict[str, int] = {}
        # term -> set of doc_ids containing it (postings list)
        self._postings: Dict[str, set[str]] = {}
        # running sum of document lengths, for avgdl
        self._total_len: int = 0

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    @property
    def num_docs(self) -> int:
        """Number of documents currently indexed."""
        return len(self._tf)

    @property
    def avgdl(self) -> float:
        """Average document length in tokens (0.0 if empty)."""
        if not self._tf:
            return 0.0
        return self._total_len / len(self._tf)

    def add(self, doc_id: str, text: str) -> None:
        """Index ``text`` under ``doc_id``, replacing any existing entry.

        Args:
            doc_id: Unique document identifier.
            text: Raw document text.
        """
        if doc_id in self._tf:
            self.remove(doc_id)

        tokens = tokenize(text)
        tf: Dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        self._tf[doc_id] = tf
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        for term in tf:
            self._postings.setdefault(term, set()).add(doc_id)

    def remove(self, doc_id: str) -> bool:
        """Remove ``doc_id`` from the index.

        Args:
            doc_id: Identifier to remove.

        Returns:
            True if the document existed and was removed, else False.
        """
        if doc_id not in self._tf:
            return False
        tf = self._tf.pop(doc_id)
        self._total_len -= self._doc_len.pop(doc_id)
        for term in tf:
            docs = self._postings.get(term)
            if docs is not None:
                docs.discard(doc_id)
                if not docs:
                    del self._postings[term]
        return True

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def idf(self, term: str) -> float:
        """Return the BM25 IDF of ``term`` against the current corpus."""
        n_q = len(self._postings.get(term, ()))
        if n_q == 0:
            return 0.0
        n_docs = len(self._tf)
        return math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))

    def _score_doc(self, doc_id: str, query_terms: Iterable[str]) -> float:
        """Compute the BM25 score of a single document for ``query_terms``."""
        tf = self._tf[doc_id]
        doc_len = self._doc_len[doc_id]
        avgdl = self.avgdl or 1.0
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf(term)
            denom = f + self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
            score += idf * (f * (self.k1 + 1.0)) / denom
        return score

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-``k`` documents for ``query`` by BM25 score.

        Only documents sharing at least one term with the query are scored,
        which keeps querying fast even for large corpora.

        Args:
            query: Raw query string.
            k: Maximum number of results to return.

        Returns:
            A list of ``(doc_id, score)`` tuples sorted by descending score.
            Ties are broken by ``doc_id`` for deterministic output.
        """
        if k <= 0 or not self._tf:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        # Gather the candidate set from postings -- the union of all docs
        # that contain at least one query term.
        candidates: set[str] = set()
        unique_terms = set(query_terms)
        for term in unique_terms:
            candidates |= self._postings.get(term, set())

        scored = [
            (doc_id, self._score_doc(doc_id, unique_terms))
            for doc_id in candidates
        ]
        scored = [(d, s) for d, s in scored if s > 0.0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def to_state(self) -> dict:
        """Serialize the index into JSON-friendly plain Python structures."""
        return {
            "k1": self.k1,
            "b": self.b,
            "tf": self._tf,
            "doc_len": self._doc_len,
            "postings": {t: sorted(d) for t, d in self._postings.items()},
            "total_len": self._total_len,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BM25Index":
        """Rebuild an index from a dict produced by :meth:`to_state`."""
        idx = cls(k1=state["k1"], b=state["b"])
        idx._tf = {d: dict(tf) for d, tf in state["tf"].items()}
        idx._doc_len = dict(state["doc_len"])
        idx._postings = {t: set(docs) for t, docs in state["postings"].items()}
        idx._total_len = int(state["total_len"])
        return idx
