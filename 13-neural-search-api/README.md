# Neural Search API

A production-grade hybrid vector-search microservice. The hard core --
the approximate-nearest-neighbor index, the lexical index, and the rank
fusion -- is implemented **from scratch** in pure Python / NumPy. Pretrained
sentence-transformer embeddings are wired in only as one *optional*
component; the service runs end to end with zero external models.

## What I implemented from scratch

Every algorithm below is real, not a wrapper around faiss / hnswlib /
elasticsearch / langchain.

- **HNSW approximate-nearest-neighbor index** -- [`core/ann.py`](core/ann.py).
  A multi-layer Hierarchical Navigable Small World graph: exponential-decay
  layer assignment (`l = floor(-ln(U) * mL)`), greedy descent through the
  upper layers, best-first beam search of width `ef` on the base layer
  (Algorithm 2 from Malkov & Yashunin), and the **neighbor-selection
  diversity heuristic** (Algorithm 4) that keeps the graph navigable instead
  of just connecting nearest points. Tunable `M`, `ef_construction`,
  `ef_search`, `ml`. Includes `brute_force_search` as exact ground truth.
- **BM25 lexical index** -- [`core/bm25.py`](core/bm25.py). From-scratch
  tokenizer, per-document term frequencies, postings lists, probabilistic
  IDF (`ln(1 + (N - n + 0.5)/(n + 0.5))`), and full Okapi BM25 scoring with
  `k1` saturation and `b` document-length normalization against `avgdl`.
  Candidate generation touches only documents sharing a query term.
- **Reciprocal-rank fusion** -- [`core/fusion.py`](core/fusion.py). Fuses the
  semantic and lexical result lists by rank position
  (`sum_l weight_l / (k + rank_l)`), with optional per-list weights. A
  min-max score-fusion baseline is included for comparison.
- **Deterministic embedder** -- [`core/embed.py`](core/embed.py). A signed
  hashing-trick bag-of-words embedder (BLAKE2b-hashed buckets, sub-linear TF
  weighting, L2 normalization) so the whole service is reproducible with no
  model download. An optional `SentenceTransformerEmbedder` is imported
  lazily behind `try/except`.
- **Hybrid retriever** -- [`core/retriever.py`](core/retriever.py). Owns the
  doc-id <-> internal-index mapping, runs both indexes, fuses them, applies
  metadata filtering and offset/limit pagination, and supports replace +
  soft-delete.
- **Persistence** -- [`core/store.py`](core/store.py). Saves/loads the whole
  index: dense vectors as `.npy`, graph topology + BM25 state + documents as
  JSON, with a versioned manifest.

The API layer ([`app.py`](app.py)) and the HTML demo
([`static/index.html`](static/index.html)) are thin shells over this core.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Then open <http://localhost:8000/> for the search demo (click *load sample
docs*), or hit the JSON API directly. Interactive API docs are at
<http://localhost:8000/docs>.

With Docker:

```bash
docker build -t neural-search .
docker run -p 8000:8000 neural-search
```

Optional pretrained embeddings: uncomment `sentence-transformers` in
`requirements.txt`, install it, and start with
`NEURAL_SEARCH_PRETRAINED=1`. Otherwise the from-scratch hashing embedder is
used.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness plus stats (`num_docs`, embedder, dim, ANN size). |
| `POST` | `/index` | Batch-index documents. Body: `{"documents": [{"id", "text", "metadata"}]}`. Duplicate ids in one batch are rejected (422). Re-indexing an existing id replaces it. |
| `POST` | `/search` | Hybrid search. Body: `{"query", "k", "filter", "offset", "limit"}`. Returns `{query, total, offset, limit, results:[{id, score, text, metadata}]}`. |
| `GET` | `/doc/{id}` | Fetch one document. 404 if missing. |
| `DELETE` | `/doc/{id}` | Delete one document. 404 if missing. |
| `POST` | `/persist?directory=...` | Save the index to disk. |
| `GET` | `/` | Static HTML search demo. |

Example:

```bash
curl -X POST localhost:8000/index -H 'content-type: application/json' -d '{
  "documents": [
    {"id": "1", "text": "vector databases power semantic search", "metadata": {"category": "tech"}},
    {"id": "2", "text": "the cat sat on the mat", "metadata": {"category": "animal"}}
  ]
}'

curl -X POST localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "semantic search engine", "k": 5, "filter": {"category": "tech"}
}'
```

Validation: bad bodies return `422` (pydantic); missing documents return
`404`.

## Verify

```bash
pip install -r requirements.txt
pytest -q
```

The suite proves the core, not just the plumbing:

- `tests/test_ann_recall.py` -- builds an HNSW index over 1000 random
  48-dim vectors and asserts **recall@10 >= 0.7** against brute-force exact
  NN, plus self-retrieval and edge cases.
- `tests/test_bm25.py` -- asserts BM25 ranks the right document first,
  rewards rare terms (IDF), and applies length normalization.
- `tests/test_fusion.py` -- checks RRF produces the exact hand-computed
  fused order and scores.
- `tests/test_retriever.py` -- hybrid search, metadata filtering,
  pagination, replace/delete, and a save/load round-trip.
- `tests/test_api.py` -- FastAPI `TestClient`: index -> search -> delete,
  filtering, and 404/422 handling.

See [`EXPLAINER.md`](EXPLAINER.md) for the HNSW algorithm and RRF math in
depth.
