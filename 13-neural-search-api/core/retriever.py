"""Hybrid retriever: dense ANN + sparse BM25 fused with RRF.

This module ties the from-scratch core together into a single document
store with hybrid retrieval, metadata filtering, and pagination:

* Documents carry an ``id``, ``text``, and arbitrary ``metadata`` dict.
* On indexing, each document is embedded (dense vector -> HNSW) and tokenized
  (BM25 postings).
* On search, the query hits both indexes; the two ranked lists are fused
  with reciprocal-rank fusion, then optionally filtered by metadata and
  sliced for pagination.

The retriever owns the mapping between human ``doc_id`` strings and the
internal integer indices used by the HNSW graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .ann import HNSWIndex
from .bm25 import BM25Index
from .embed import HashingEmbedder
from .fusion import reciprocal_rank_fusion


@dataclass
class Document:
    """A stored document.

    Attributes:
        id: Unique, caller-supplied identifier.
        text: The raw text that was indexed.
        metadata: Arbitrary JSON-serializable key/value pairs used for
            filtering.
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """A single search result.

    Attributes:
        id: The matching document id.
        score: The fused relevance score (higher is better).
        text: The document text.
        metadata: The document metadata.
    """

    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


# A metadata filter is a function from a metadata dict to bool.
MetadataFilter = Callable[[Dict[str, Any]], bool]


def make_filter(spec: Optional[Dict[str, Any]]) -> Optional[MetadataFilter]:
    """Build a metadata filter predicate from a simple equality spec.

    The spec is a flat ``{key: value}`` dict; a document matches if every
    key is present in its metadata and compares equal. If a spec value is a
    list, membership is tested (``metadata[key] in value``).

    Args:
        spec: Equality/membership spec, or None for "match everything".

    Returns:
        A predicate, or None if ``spec`` is falsy.
    """
    if not spec:
        return None

    def predicate(meta: Dict[str, Any]) -> bool:
        for key, want in spec.items():
            if key not in meta:
                return False
            have = meta[key]
            if isinstance(want, list):
                if have not in want:
                    return False
            elif have != want:
                return False
        return True

    return predicate


class HybridRetriever:
    """A hybrid dense+sparse document retriever.

    Args:
        embedder: Any object implementing the embedder protocol. Defaults to
            the from-scratch :class:`HashingEmbedder`.
        M: HNSW neighbor budget.
        ef_construction: HNSW build-time beam width.
        ef_search: HNSW query-time beam width.
        bm25_k1: BM25 term-frequency saturation.
        bm25_b: BM25 length-normalization strength.
        rrf_k: RRF damping constant.
        semantic_weight: RRF weight applied to the dense (ANN) list.
        lexical_weight: RRF weight applied to the sparse (BM25) list.
    """

    def __init__(
        self,
        embedder: Optional[HashingEmbedder] = None,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.embedder = embedder or HashingEmbedder(dim=256)
        self.ann = HNSWIndex(
            dim=self.embedder.dim,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self.rrf_k = int(rrf_k)
        self.semantic_weight = float(semantic_weight)
        self.lexical_weight = float(lexical_weight)

        # doc_id -> Document
        self._docs: Dict[str, Document] = {}
        # internal HNSW index -> doc_id  (HNSW indices are never reused)
        self._idx_to_id: Dict[int, str] = {}
        # doc_id -> internal HNSW index
        self._id_to_idx: Dict[str, int] = {}
        # Soft-deleted doc ids: still in the graph, hidden from results.
        self._deleted: set[str] = set()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def num_docs(self) -> int:
        """Number of live (non-deleted) documents."""
        return len(self._docs) - len(self._deleted)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs and doc_id not in self._deleted

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def add(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Index a single document, replacing any existing one with the id.

        Args:
            doc_id: Unique identifier.
            text: Document text.
            metadata: Optional metadata dict.
        """
        if doc_id in self._docs and doc_id not in self._deleted:
            # Replace: BM25 supports in-place replace; HNSW cannot delete an
            # edge cheaply, so we re-embed into a fresh internal node and
            # orphan the old one (it becomes unreachable from results).
            self._retire_internal(doc_id)

        doc = Document(id=doc_id, text=text, metadata=dict(metadata or {}))
        self._docs[doc_id] = doc
        self._deleted.discard(doc_id)

        vec = self.embedder.embed(text)
        internal = self.ann.add(vec)
        self._idx_to_id[internal] = doc_id
        self._id_to_idx[doc_id] = internal

        self.bm25.add(doc_id, text)

    def add_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Index a batch of ``{id, text, metadata}`` dicts.

        Args:
            documents: List of document dicts. ``metadata`` is optional.

        Returns:
            The number of documents indexed.
        """
        count = 0
        for d in documents:
            self.add(d["id"], d["text"], d.get("metadata"))
            count += 1
        return count

    def _retire_internal(self, doc_id: str) -> None:
        """Detach the internal HNSW mapping for a doc that is being replaced."""
        old_internal = self._id_to_idx.pop(doc_id, None)
        if old_internal is not None:
            self._idx_to_id.pop(old_internal, None)
        self.bm25.remove(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Soft-delete a document so it stops appearing in results.

        The vector remains in the HNSW graph (needed to preserve graph
        connectivity), but the id is removed from BM25 and hidden from all
        result lists.

        Args:
            doc_id: Identifier to delete.

        Returns:
            True if the document existed and was deleted, else False.
        """
        if doc_id not in self._docs or doc_id in self._deleted:
            return False
        self._deleted.add(doc_id)
        self.bm25.remove(doc_id)
        return True

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        k: int = 10,
        filter_spec: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> Tuple[List[SearchHit], int]:
        """Run a hybrid search and return a page of results.

        Args:
            query: The query string.
            k: How many fused candidates to consider before filtering. Also
                the number of items each sub-index contributes.
            filter_spec: Optional metadata equality/membership filter.
            offset: Pagination offset into the filtered result list.
            limit: Pagination page size. Defaults to ``k`` when None.
            candidate_k: How many candidates to pull from each sub-index.
                Defaults to ``max(k * 4, 50)`` so filtering and fusion have
                enough material to work with.

        Returns:
            A tuple ``(hits, total)`` where ``hits`` is the requested page and
            ``total`` is the number of results after filtering (for the
            caller to compute pagination metadata).
        """
        if not query or k <= 0 or self.num_docs == 0:
            return [], 0

        cand = candidate_k if candidate_k is not None else max(k * 4, 50)

        # Dense (semantic) results from HNSW, mapped back to doc ids.
        qvec = self.embedder.embed(query)
        ann_raw = self.ann.search(qvec, k=cand)
        ann_list: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for internal, sim in ann_raw:
            doc_id = self._idx_to_id.get(internal)
            if doc_id is None or doc_id in self._deleted or doc_id in seen:
                continue
            seen.add(doc_id)
            ann_list.append((doc_id, sim))

        # Sparse (lexical) results from BM25.
        bm25_list = [
            (doc_id, score)
            for doc_id, score in self.bm25.search(query, k=cand)
            if doc_id not in self._deleted
        ]

        # Fuse with reciprocal-rank fusion.
        fused = reciprocal_rank_fusion(
            [ann_list, bm25_list],
            k=self.rrf_k,
            weights=[self.semantic_weight, self.lexical_weight],
        )

        # Apply metadata filtering.
        predicate = make_filter(filter_spec)
        filtered: List[Tuple[str, float]] = []
        for doc_id, score in fused:
            doc = self._docs.get(doc_id)
            if doc is None:
                continue
            if predicate is not None and not predicate(doc.metadata):
                continue
            filtered.append((doc_id, score))

        total = len(filtered)

        # Paginate.
        page_size = limit if limit is not None else k
        start = max(offset, 0)
        end = start + max(page_size, 0)
        page = filtered[start:end]

        hits = [
            SearchHit(
                id=doc_id,
                score=float(score),
                text=self._docs[doc_id].text,
                metadata=self._docs[doc_id].metadata,
            )
            for doc_id, score in page
        ]
        return hits, total

    def get(self, doc_id: str) -> Optional[Document]:
        """Return a live document by id, or None if missing/deleted."""
        if doc_id in self._deleted:
            return None
        return self._docs.get(doc_id)

    # ------------------------------------------------------------------ #
    # State for persistence
    # ------------------------------------------------------------------ #
    def docs_state(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict of all document + mapping state."""
        return {
            "docs": {
                doc_id: {
                    "id": d.id,
                    "text": d.text,
                    "metadata": d.metadata,
                }
                for doc_id, d in self._docs.items()
            },
            "idx_to_id": {str(i): did for i, did in self._idx_to_id.items()},
            "id_to_idx": dict(self._id_to_idx),
            "deleted": sorted(self._deleted),
            "rrf_k": self.rrf_k,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
            "embedder_dim": self.embedder.dim,
        }

    def load_docs_state(self, state: Dict[str, Any]) -> None:
        """Restore document + mapping state from :meth:`docs_state`."""
        self._docs = {
            doc_id: Document(
                id=d["id"], text=d["text"], metadata=dict(d.get("metadata", {}))
            )
            for doc_id, d in state["docs"].items()
        }
        self._idx_to_id = {int(i): did for i, did in state["idx_to_id"].items()}
        self._id_to_idx = dict(state["id_to_idx"])
        self._deleted = set(state.get("deleted", []))
        self.rrf_k = int(state.get("rrf_k", self.rrf_k))
        self.semantic_weight = float(
            state.get("semantic_weight", self.semantic_weight)
        )
        self.lexical_weight = float(
            state.get("lexical_weight", self.lexical_weight)
        )
