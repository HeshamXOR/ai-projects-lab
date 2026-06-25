# Explainer: HNSW and Reciprocal-Rank Fusion

This document goes under the hood of the two most interesting pieces of the
from-scratch core: the **HNSW** approximate-nearest-neighbor index
([`core/ann.py`](core/ann.py)) and **reciprocal-rank fusion**
([`core/fusion.py`](core/fusion.py)).

---

## 1. HNSW: Hierarchical Navigable Small World graphs

### The problem

Exact nearest-neighbor search over `N` vectors costs `O(N·d)` per query --
fine for thousands of documents, ruinous for millions. We want sub-linear
query time while keeping recall high. HNSW (Malkov & Yashunin, 2018) does
this with a layered proximity graph that you navigate greedily.

### The structure

HNSW is a stack of graphs:

```
layer 2:        A ---------------- F            (sparse, long hops)
layer 1:    A --- C --- F --- H                 (medium density)
layer 0:  A-B-C-D-E-F-G-H-I-J-K-L-...            (every node, dense)
```

- **Layer 0** contains every point and is densely connected (`M0 = 2M`
  edges per node).
- **Higher layers** contain a geometrically shrinking subset of points and
  act like the express lanes of a skip list -- a few long hops get you into
  the right neighborhood fast.

### Layer assignment

When a point is inserted, its top layer is drawn from an exponentially
decaying distribution:

```
l = floor( -ln(U) * mL ),   U ~ Uniform(0,1),   mL = 1 / ln(M)
```

With `mL = 1/ln(M)`, the probability of reaching layer `l` falls off by a
factor of `1/M` per level, so the expected number of layers is `O(log N)`
and most points live only on layer 0. This is `_random_level()` in the code.

### Searching a single layer (Algorithm 2)

The primitive `_search_layer(query, entry_points, ef, layer)` is a
**best-first beam search**:

- A **candidate** min-heap holds frontier nodes ordered by distance to the
  query (closest popped first).
- A **result** heap (a max-heap implemented with negated distances) holds
  the best `ef` nodes found so far.
- We repeatedly expand the closest candidate's neighbors. We stop early when
  the closest remaining candidate is farther than the worst element of a
  full result set -- there is nothing better left to find.

`ef` (the beam width) is the central quality/speed knob: larger `ef` =
explore more of the graph = higher recall, slower query.

### Query (descend then beam)

`search(query, k, ef_search)`:

1. Start at the global entry point on the top layer.
2. For each layer from the top down to layer 1, greedily walk to the closest
   node using `ef = 1` (pure hill-climbing through the express lanes).
3. On layer 0, run the full beam search with width `ef = max(ef_search, k)`
   and return the `k` closest, converting cosine distance back to
   similarity.

The upper-layer descent is `O(log N)` hops; the base-layer beam search
touches `O(ef · M)` nodes. No full scan.

### Insertion and the neighbor heuristic (Algorithm 4)

Insertion reuses the same machinery: descend with `ef=1` down to the new
node's top layer, then on every layer at or below it, run the beam search
with width `ef_construction` to gather candidate neighbors and connect.

The subtle, important part is **which** neighbors to keep. Naively picking
the `M` closest produces clustered, poorly-navigable graphs. Instead
`_select_neighbors_heuristic` keeps a candidate `c` only if `c` is closer to
the new point than to any neighbor already selected:

```
keep c  iff  for all already-selected s:  dist(c, query) <= dist(c, s)
```

This favors **diverse** neighbors that point in different directions,
preserving the long-range connectivity that makes greedy search land in the
right place. When a node exceeds its edge budget after gaining a reciprocal
link, `_prune` re-runs the same heuristic to trim back to `M` (or `M0` on
layer 0).

### Distances

All vectors are L2-normalized on insertion, so cosine similarity is a dot
product and cosine **distance** is `1 - dot`. This keeps the inner loop to a
single `np.dot`.

### Why the recall test matters

`tests/test_ann_recall.py` is the proof the graph actually works: it builds
an index over 1000 random vectors and checks that the approximate top-10
overlaps the brute-force exact top-10 by at least 70% on average. A broken
graph (bad heuristic, wrong stop condition) tanks this number immediately.

---

## 2. Reciprocal-Rank Fusion

### The problem

We have two result lists for the same query:

- a **semantic** list from HNSW, scored by cosine similarity in `[-1, 1]`;
- a **lexical** list from BM25, scored by an unbounded relevance score.

These scores live on completely different scales, so we cannot just add
them. We could min-max normalize and blend (the `weighted_score_fusion`
baseline does exactly that), but normalization is fragile -- one outlier
score warps the whole list.

### The fix: fuse on rank, not score

RRF throws away the raw scores and uses only **rank position**. For an item
`d` at rank `r_l(d)` (1-indexed) in list `l`:

```
RRF(d) = sum over lists l of   weight_l / (k + r_l(d))
```

Items missing from a list contribute nothing for that list.

### Intuition for `k`

The constant `k` (default 60, the value from the original Cormack et al.
paper) damps the head of the distribution. Compare the contribution of
rank 1 vs rank 2:

- with `k = 0`:  `1/1 = 1.0` vs `1/2 = 0.5`  -- rank 1 is worth 2x rank 2.
- with `k = 60`: `1/61 ≈ 0.0164` vs `1/62 ≈ 0.0161` -- nearly equal.

A larger `k` means "being in the top handful matters; the exact position
within it matters less." This is what lets an item that appears at a
respectable rank in **both** lists overtake an item sitting at #1 in only
one list -- which is exactly the behavior we want from a hybrid engine, and
exactly what `tests/test_fusion.py` asserts with hand-computed values:

```
list A: [x, y, z]      list B: [y, x, w]
x = 1/61 + 1/62        (in both)
y = 1/62 + 1/61        (in both, ties x)
z = 1/63               (one list)
w = 1/63               (one list)
=> x, y rank above z, w
```

### Weights

Per-list weights let you bias the blend -- e.g. `weights=[1.0, 1.5]` to
trust lexical matches a bit more on a keyword-heavy corpus. The retriever
exposes these as `semantic_weight` / `lexical_weight`.

---

## Putting it together

`core/retriever.py` runs both indexes for a query, maps HNSW internal
indices back to document ids, fuses the two lists with RRF, applies the
metadata filter, and paginates. The result is a search engine that catches
both *"semantic search engine"* matching *"vector databases power semantic
search"* (dense) **and** an exact rare-keyword hit (lexical) -- without
either signal drowning the other.
