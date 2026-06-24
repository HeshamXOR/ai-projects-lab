"""RAG engine: chunk a PDF, embed, retrieve (hybrid), and answer with citations.

The retrieval here is the point of this project. Instead of calling a vector-DB
library, it runs a **from-scratch hybrid retriever**:

  * a from-scratch **HNSW** graph index for fast approximate semantic search
    (core/hnsw.py),
  * a from-scratch **BM25** keyword index for exact-term precision (core/bm25.py),
  * **reciprocal-rank fusion** to combine them (core/fusion.py).

The embedding model and (optional) LLM are the only pretrained components; the
search algorithm is implemented by hand. The answerer stays extractive by
default (works with no API key) and upgrades to generative if a key is set.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.bm25 import BM25
from core.fusion import reciprocal_rank_fusion
from core.hnsw import HNSW


@dataclass
class Chunk:
    text: str
    page: int


@dataclass
class Answer:
    text: str
    sources: List[Chunk]


def extract_pdf_chunks(pdf_path: str, chunk_chars: int = 800, overlap: int = 120) -> List[Chunk]:
    """Read a PDF into overlapping text chunks tagged with their page number."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    chunks: List[Chunk] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_chars]
            chunks.append(Chunk(text=piece, page=page_idx))
            if start + chunk_chars >= len(text):
                break
            start += chunk_chars - overlap
    return chunks


class RagEngine:
    """Embedding model + a from-scratch hybrid index (HNSW vector + BM25)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self._hnsw: Optional[HNSW] = None
        self._bm25: Optional[BM25] = None
        self._dim: Optional[int] = None

    def index_pdf(self, pdf_path: str) -> int:
        """Build the hybrid index for one PDF. Returns the number of chunks."""
        self.chunks = extract_pdf_chunks(pdf_path)
        if not self.chunks:
            self._hnsw = self._bm25 = None
            return 0
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        # 1) vector index: from-scratch HNSW
        self._dim = embeddings.shape[1]
        self._hnsw = HNSW(dim=self._dim, M=16, ef_construction=200, ef_search=64)
        self._hnsw.add_batch(embeddings)
        # 2) keyword index: from-scratch BM25
        self._bm25 = BM25()
        self._bm25.index(texts)
        return len(self.chunks)

    def retrieve(self, question: str, k: int = 4, hybrid: bool = True) -> List[Chunk]:
        """Hybrid retrieval: HNSW semantic ⊕ BM25 keyword, fused with RRF."""
        if self._hnsw is None or not self.chunks:
            return []
        q_emb = self.model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        # semantic candidates from the from-scratch HNSW graph
        vec_hits = self._hnsw.search(q_emb, k=max(k * 2, 8))
        vec_ranked = [(doc_id, -dist) for doc_id, dist in vec_hits]  # closer = better

        if not hybrid or self._bm25 is None:
            return [self.chunks[i] for i, _ in vec_ranked[:k]]

        # keyword candidates from from-scratch BM25
        kw_ranked = self._bm25.search(question, k=max(k * 2, 8))
        # fuse the two rankings
        fused = reciprocal_rank_fusion([vec_ranked, kw_ranked], k=60, top_n=k)
        return [self.chunks[i] for i, _ in fused]

    def answer(self, question: str, k: int = 4) -> Answer:
        hits = self.retrieve(question, k=k)
        if not hits:
            return Answer(text="No document indexed yet — upload a PDF first.", sources=[])

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            text = self._answer_with_llm(question, hits, api_key)
        else:
            text = self._answer_extractive(question, hits)
        return Answer(text=text, sources=hits)

    # --- extractive (default, no key needed) ---
    def _answer_extractive(self, question: str, hits: List[Chunk]) -> str:
        """Pick the sentences from retrieved chunks most similar to the question."""
        sentences: List[tuple] = []  # (sentence, page)
        for c in hits:
            for s in re.split(r"(?<=[.!?])\s+", c.text):
                s = s.strip()
                if len(s) > 30:
                    sentences.append((s, c.page))
        if not sentences:
            joined = " ".join(c.text for c in hits)[:500]
            return f"{joined}\n\n(Extractive answer — set OPENAI_API_KEY for a generated answer.)"

        embs = self.model.encode(
            [s for s, _ in sentences], convert_to_numpy=True, normalize_embeddings=True
        )
        q = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores = embs @ q[0]
        order = np.argsort(-scores)[:3]
        picked = [sentences[i] for i in sorted(order)]
        body = " ".join(s for s, _ in picked)
        pages = sorted({p for _, p in picked})
        cite = ", ".join(f"p.{p}" for p in pages)
        return f"{body}\n\n— Based on {cite}. (Extractive mode; set OPENAI_API_KEY for a generated answer.)"

    # --- generative (optional upgrade) ---
    def _answer_with_llm(self, question: str, hits: List[Chunk], api_key: str) -> str:
        import httpx

        context = "\n\n".join(f"[p.{c.page}] {c.text}" for c in hits)
        base_url = os.environ.get("OPENFORGE_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENFORGE_MODEL", "gpt-4o-mini")
        prompt = (
            "Answer the question using ONLY the context below. Cite page numbers "
            "in the form (p.N). If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # fall back rather than break the demo
            return self._answer_extractive(question, hits) + f"\n\n(LLM call failed: {e})"
