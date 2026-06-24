# 📄 RAG Document Q&A — with a from-scratch vector index

Upload a PDF, ask questions, get answers **grounded in the document** with page citations. Unlike a typical RAG demo, the retrieval engine here is **built from scratch**, not imported.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **HNSW** approximate-nearest-neighbor index (the algorithm inside FAISS/Qdrant/pgvector) — `core/hnsw.py`
- **BM25** keyword ranking (the core of Elasticsearch) — `core/bm25.py`
- **Reciprocal Rank Fusion** to combine semantic + keyword search — `core/fusion.py`

The embedding model and optional answer-LLM are the only pretrained components. See [EXPLAINER.md](EXPLAINER.md) for the full how-and-why.

## Pipeline

```
PDF → page-tagged overlapping chunks → embed
   ├─ HNSW graph index   (semantic search, ~O(log N))
   └─ BM25 keyword index (exact-term precision)
        → reciprocal-rank fusion → top-k → cited answer
```

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860 (+ public gradio.live link)
```

Default answers are extractive (no API key needed). Set `OPENAI_API_KEY` for generated answers.

## Verify the engine

```bash
pytest -q          # recall@10 ≥ 0.90 vs exact brute force; BM25 + RRF correctness
python bench.py    # build time, latency, speedup vs brute force (and FAISS if installed)
```

## Limitations

- Pure-Python HNSW is for *understanding* the algorithm; FAISS will be faster in raw wall-clock (C++/SIMD). The win demonstrated here is algorithmic scaling vs. brute force.
- Scanned/image PDFs need OCR (not included).
