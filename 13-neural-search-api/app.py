"""FastAPI application exposing the hybrid neural-search service.

Endpoints:

* ``POST   /index``      -- batch-index documents.
* ``POST   /search``     -- hybrid search with filtering + pagination.
* ``DELETE /doc/{id}``   -- delete a single document.
* ``GET    /doc/{id}``   -- fetch a single document.
* ``GET    /health``     -- liveness + basic stats.
* ``POST   /persist``    -- save the index to disk.
* ``GET    /``           -- the static HTML search demo.

The service holds a single in-process :class:`HybridRetriever`. All heavy
lifting (ANN graph, BM25, fusion) is the from-scratch core in :mod:`core`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from core import store
from core.embed import get_embedder
from core.retriever import HybridRetriever

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INDEX_DIR = os.environ.get("NEURAL_SEARCH_INDEX_DIR", "./index_data")
EMBED_DIM = int(os.environ.get("NEURAL_SEARCH_EMBED_DIM", "256"))
PREFER_PRETRAINED = os.environ.get("NEURAL_SEARCH_PRETRAINED", "0") == "1"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class DocumentIn(BaseModel):
    """A document to index."""

    id: str = Field(..., min_length=1, description="Unique document id.")
    text: str = Field(..., min_length=1, description="Document text.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary filterable metadata."
    )


class IndexRequest(BaseModel):
    """Batch-index request body."""

    documents: List[DocumentIn] = Field(
        ..., min_length=1, description="Documents to index."
    )

    @field_validator("documents")
    @classmethod
    def _unique_ids(cls, docs: List[DocumentIn]) -> List[DocumentIn]:
        """Reject batches containing duplicate ids."""
        ids = [d.id for d in docs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate document ids in the same batch")
        return docs


class IndexResponse(BaseModel):
    """Response for a successful index call."""

    indexed: int = Field(..., description="Number of documents indexed.")
    total_docs: int = Field(..., description="Live documents after indexing.")


class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(..., min_length=1, description="Query string.")
    k: int = Field(10, ge=1, le=1000, description="Candidates to consider.")
    filter: Optional[Dict[str, Any]] = Field(
        None, description="Metadata equality/membership filter."
    )
    offset: int = Field(0, ge=0, description="Pagination offset.")
    limit: Optional[int] = Field(
        None, ge=1, le=1000, description="Page size (defaults to k)."
    )


class SearchHitOut(BaseModel):
    """A single search hit."""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response with pagination metadata."""

    query: str
    total: int = Field(..., description="Total matches after filtering.")
    offset: int
    limit: int
    results: List[SearchHitOut]


class DeleteResponse(BaseModel):
    """Response for a delete call."""

    id: str
    deleted: bool


class HealthResponse(BaseModel):
    """Liveness + stats."""

    status: str
    num_docs: int
    embedder: str
    embed_dim: int
    ann_size: int


# --------------------------------------------------------------------------- #
# Application + state
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Neural Search API",
    version="1.0.0",
    description=(
        "Hybrid vector search (HNSW + BM25 + RRF) with a from-scratch core."
    ),
)


def _build_retriever() -> HybridRetriever:
    """Construct a retriever, loading from disk if a saved index exists."""
    if os.path.isfile(os.path.join(INDEX_DIR, "manifest.json")):
        try:
            return store.load(INDEX_DIR)
        except Exception:
            # Corrupt / incompatible on-disk index: start fresh rather than
            # refusing to boot.
            pass
    embedder = get_embedder(prefer_pretrained=PREFER_PRETRAINED, dim=EMBED_DIM)
    return HybridRetriever(embedder=embedder)


retriever: HybridRetriever = _build_retriever()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Return liveness and basic index statistics."""
    return HealthResponse(
        status="ok",
        num_docs=retriever.num_docs,
        embedder=getattr(retriever.embedder, "name", "unknown"),
        embed_dim=retriever.embedder.dim,
        ann_size=retriever.ann.size,
    )


@app.post("/index", response_model=IndexResponse, tags=["index"])
def index_documents(req: IndexRequest) -> IndexResponse:
    """Batch-index documents.

    Existing documents with the same id are replaced.
    """
    payload = [
        {"id": d.id, "text": d.text, "metadata": d.metadata}
        for d in req.documents
    ]
    count = retriever.add_batch(payload)
    return IndexResponse(indexed=count, total_docs=retriever.num_docs)


@app.post("/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    """Run a hybrid (semantic + lexical) search."""
    hits, total = retriever.search(
        query=req.query,
        k=req.k,
        filter_spec=req.filter,
        offset=req.offset,
        limit=req.limit,
    )
    limit = req.limit if req.limit is not None else req.k
    return SearchResponse(
        query=req.query,
        total=total,
        offset=req.offset,
        limit=limit,
        results=[
            SearchHitOut(
                id=h.id, score=h.score, text=h.text, metadata=h.metadata
            )
            for h in hits
        ],
    )


@app.get("/doc/{doc_id}", response_model=DocumentIn, tags=["index"])
def get_document(doc_id: str) -> DocumentIn:
    """Fetch a single live document by id, or 404 if absent."""
    doc = retriever.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"doc {doc_id!r} not found")
    return DocumentIn(id=doc.id, text=doc.text, metadata=doc.metadata)


@app.delete("/doc/{doc_id}", response_model=DeleteResponse, tags=["index"])
def delete_document(doc_id: str) -> DeleteResponse:
    """Delete a document by id. Returns 404 if it does not exist."""
    ok = retriever.delete(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"doc {doc_id!r} not found")
    return DeleteResponse(id=doc_id, deleted=True)


@app.post("/persist", tags=["ops"])
def persist(directory: str = Query(default=INDEX_DIR)) -> JSONResponse:
    """Persist the current index to ``directory`` on disk."""
    store.save(retriever, directory)
    return JSONResponse(
        {"status": "saved", "directory": directory, "num_docs": retriever.num_docs}
    )


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    """Serve the static HTML search demo."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Mount the static directory for assets (created at write time).
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
