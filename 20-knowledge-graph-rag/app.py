"""knowledge-graph-rag — entity-graph-augmented RAG (FastAPI service).

Ingest documents, extract entities + relations, build an in-memory knowledge
graph, and answer questions with graph-augmented retrieval. All of the hard
parts — extraction, the graph, and the retriever — are implemented from scratch
in ``core/`` (no networkx, no spaCy). State is held in memory for the process
lifetime.

Endpoints:
    GET  /                  service banner
    GET  /health            liveness probe
    POST /ingest            add text/documents -> chunk, extract, index, graph
    POST /ask               (graph-augmented) retrieval + extractive answer
    GET  /graph             node/edge summary
    GET  /graph/path        shortest path between two entities
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from core.extract import extract
from core.graph import KnowledgeGraph
from core.retrieval import Retriever, assemble_answer

app = FastAPI(
    title="knowledge-graph-rag",
    description="Entity-graph-augmented RAG with from-scratch extraction, KG, and retrieval.",
    version="1.0.0",
)


class _State:
    """Process-lifetime in-memory state: the graph and the passage retriever."""

    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        self.retriever = Retriever(passages=[], graph=self.graph).fit()
        self.num_docs = 0

    def reset(self) -> None:
        self.__init__()


STATE = _State()


def chunk_text(text: str, min_len: int = 15) -> List[str]:
    """Split text into one passage per sentence (transparent chunking)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


# ----------------------------------------------------------------- schemas
class IngestRequest(BaseModel):
    """Either ``text`` (one blob) or ``documents`` (a list) must be provided."""

    text: Optional[str] = Field(default=None, description="A single text blob.")
    documents: Optional[List[str]] = Field(
        default=None, description="A list of documents to ingest."
    )

    @model_validator(mode="after")
    def _require_content(self) -> "IngestRequest":
        if not self.text and not self.documents:
            raise ValueError("provide either 'text' or 'documents'")
        return self


class IngestResponse(BaseModel):
    passages_added: int
    total_passages: int
    entities: int
    edges: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_graph: bool = Field(default=True, description="Enable graph expansion.")
    k: int = Field(default=3, ge=1, le=20, description="Top-k passages.")
    hops: int = Field(default=1, ge=1, le=4, description="Graph expansion hops.")


class PassageOut(BaseModel):
    index: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    passages: List[PassageOut]
    entities_expanded: List[str]
    path: Optional[List[str]] = None
    used_graph: bool


class GraphSummary(BaseModel):
    num_nodes: int
    num_edges: int
    nodes: List[str]
    edges: List[Dict[str, object]]


class PathResponse(BaseModel):
    src: str
    dst: str
    path: Optional[List[str]]
    found: bool


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root() -> Dict[str, object]:
    """Service banner with a one-line description and endpoint list."""
    return {
        "service": "knowledge-graph-rag",
        "description": "Entity-graph-augmented RAG (from-scratch extraction, KG, retrieval).",
        "endpoints": ["/health", "/ingest", "/ask", "/graph", "/graph/path"],
    }


@app.get("/health")
def health() -> Dict[str, object]:
    """Liveness probe with current state sizes."""
    return {
        "status": "ok",
        "passages": len(STATE.retriever.passages),
        "nodes": STATE.graph.num_nodes(),
        "edges": STATE.graph.num_edges(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Chunk inputs into passages, extract entities/relations, update KG + index."""
    blobs: List[str] = []
    if req.text:
        blobs.append(req.text)
    if req.documents:
        blobs.extend(req.documents)

    new_passages: List[str] = []
    for blob in blobs:
        new_passages.extend(chunk_text(blob))
        result = extract(blob)
        for ent in result.entities:
            STATE.graph.add_entity(ent)
        for head, rel, tail in result.triples:
            STATE.graph.add_relation(head, rel, tail, weight=2.0)
        for (a, b), count in result.cooccurrence.items():
            STATE.graph.add_relation(a, "co_occurs", b, weight=float(count))

    if not new_passages:
        raise HTTPException(status_code=400, detail="No usable passages found in input.")

    STATE.retriever.add_passages(new_passages)
    STATE.num_docs += len(blobs)
    return IngestResponse(
        passages_added=len(new_passages),
        total_passages=len(STATE.retriever.passages),
        entities=STATE.graph.num_nodes(),
        edges=STATE.graph.num_edges(),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Run (graph-augmented) retrieval and return an extractive answer."""
    if not STATE.retriever.passages:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    expanded: List[str] = []
    if req.use_graph:
        scored, expanded = STATE.retriever.graph_augmented_retrieve(
            req.question, k=req.k, hops=req.hops
        )
    else:
        scored = STATE.retriever.retrieve(req.question, k=req.k)

    answer = assemble_answer(req.question, scored)

    # if exactly two seed entities matched, surface the connecting path
    path: Optional[List[str]] = None
    seeds = STATE.graph.match_seeds(req.question)
    if len(seeds) >= 2:
        path = STATE.graph.shortest_path(seeds[0], seeds[1])

    return AskResponse(
        answer=answer,
        passages=[PassageOut(index=s.index, text=s.text, score=round(s.score, 4)) for s in scored],
        entities_expanded=expanded,
        path=path,
        used_graph=req.use_graph,
    )


@app.get("/graph", response_model=GraphSummary)
def graph_summary() -> GraphSummary:
    """Return a summary of the knowledge graph (nodes and edges)."""
    edges = [
        {"head": h, "relation": r, "tail": t, "weight": round(w, 3)}
        for (h, r, t, w) in STATE.graph.edges()
    ]
    return GraphSummary(
        num_nodes=STATE.graph.num_nodes(),
        num_edges=STATE.graph.num_edges(),
        nodes=STATE.graph.nodes,
        edges=edges,
    )


@app.get("/graph/path", response_model=PathResponse)
def graph_path(
    src: str = Query(..., min_length=1),
    dst: str = Query(..., min_length=1),
) -> PathResponse:
    """Return the shortest path between two entities (undirected traversal)."""
    if src not in STATE.graph.nodes:
        raise HTTPException(status_code=404, detail=f"Unknown source entity: {src!r}")
    if dst not in STATE.graph.nodes:
        raise HTTPException(status_code=404, detail=f"Unknown destination entity: {dst!r}")
    path = STATE.graph.shortest_path(src, dst)
    return PathResponse(src=src, dst=dst, path=path, found=path is not None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
