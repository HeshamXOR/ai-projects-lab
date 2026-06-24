# EXPLAINER — RAG Doc Q&A: the retrieval is the project

## What I implemented from scratch

A **hybrid retrieval engine** — the part a typical RAG demo imports from a library:

1. **HNSW** (Hierarchical Navigable Small World) approximate-nearest-neighbor index — the algorithm behind FAISS, Qdrant, Weaviate, and pgvector. `core/hnsw.py`.
2. **BM25** probabilistic keyword ranking — the backbone of Elasticsearch. `core/bm25.py`.
3. **Reciprocal Rank Fusion** to combine semantic + keyword results. `core/fusion.py`.

The only pretrained pieces are the sentence-embedding model and the optional answer LLM. The *search* is mine.

## Why HNSW (and how it works)

Brute-force nearest-neighbor search compares the query to every vector: O(N) per query. That's fine for a 50-page PDF, useless for millions of chunks. HNSW gets ~O(log N) by building a **multi-layer navigable graph**:

- **Layers**: layer 0 contains every node; each higher layer keeps an exponentially smaller random subset (like a skip list). A node's top layer is drawn from a geometric distribution.
- **Search**: start at the single entry point on the top layer, greedily hop to the closest neighbor you can see, and when you can't get closer, drop down a layer and continue. You "zoom in" — coarse navigation up high, fine-grained at the bottom.
- **Insertion**: search for the new node's nearest neighbors at each layer and connect them, capped at `M` edges per node.
- **The subtle part — neighbor selection**: I don't just link the M closest nodes. I use the paper's heuristic (Algorithm 4): prefer a candidate only if it's closer to the new node than to any already-chosen neighbor. This keeps edges *diverse* so the graph stays globally navigable instead of forming isolated clumps. Getting this wrong tanks recall, which is why it's the most carefully-commented function.

Two knobs trade recall for speed: `ef_construction` (graph quality at build time) and `ef_search` (candidate breadth at query time).

## Why also BM25, and why fuse

Vectors capture *meaning* but miss exact tokens — product codes, names, numbers, rare jargon. BM25 nails those but misses paraphrase. Running both and fusing gets the best of each.

The fusion problem: vector scores (cosine) and BM25 scores live on totally different scales, so you can't just add them. **Reciprocal Rank Fusion** ignores the scores and uses only *rank*: `score(d) = Σ 1/(k + rank_i(d))`. A chunk ranked highly by both retrievers wins. Simple, robust, and what production systems actually use.

## How it fits together (`rag.py`)

```
PDF → overlapping page-tagged chunks → embed each chunk
                                      ├─→ HNSW index (vectors)
                                      └─→ BM25 index (tokens)

question → embed → HNSW.search ┐
         → tokens → BM25.search ┘ → RRF fuse → top-k chunks → cited answer
```

## Proof it works

`tests/test_core.py`:
- **`test_hnsw_recall_vs_brute_force`** asserts recall@10 ≥ 0.90 against exact brute-force ground truth on a fixed 500×32 dataset. (Run `pytest` to see the actual figure on your machine.)
- **`test_hnsw_finds_exact_self`** — querying an indexed vector returns itself at distance ≈ 0.
- BM25 and RRF have their own correctness tests.

`bench.py` reports build time, query latency, **speedup vs. brute force**, and recall on a larger 5000×64 set — and compares to FAISS if it's installed. Run it on Lightning to fill in the README's numbers.

## Honest limitations

- Pure-Python HNSW is correct but not fast in wall-clock terms vs. FAISS's C++/SIMD — the win shown here is *algorithmic* (scaling vs. brute force), not raw constant-factor speed. That's the honest framing: I implemented the algorithm to understand it, not to beat a hand-optimized library.
- No deletion/updating of nodes (insert + search only).
- Single-PDF scope in the demo; the index code itself is corpus-agnostic.
