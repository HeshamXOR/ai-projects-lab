"""RAG engine: chunk a PDF, embed, retrieve, and answer with citations.

Kept separate from the Gradio UI so it can be tested and reused. The design
goal is "works on a free CPU Studio with no API key" — so the default answerer
is *extractive* (it stitches together the most relevant retrieved sentences and
cites their pages). If an OpenAI-compatible key is configured, it upgrades to a
generative answer. Either way you always get cited, grounded output.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


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
    """Holds the embedding model and a per-document index."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None

    def index_pdf(self, pdf_path: str) -> int:
        """Build the index for one PDF. Returns the number of chunks."""
        self.chunks = extract_pdf_chunks(pdf_path)
        if not self.chunks:
            self._embeddings = None
            return 0
        texts = [c.text for c in self.chunks]
        self._embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return len(self.chunks)

    def retrieve(self, question: str, k: int = 4) -> List[Chunk]:
        if self._embeddings is None or not self.chunks:
            return []
        q = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        # cosine similarity == dot product on normalized vectors
        scores = (self._embeddings @ q[0])
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]

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
