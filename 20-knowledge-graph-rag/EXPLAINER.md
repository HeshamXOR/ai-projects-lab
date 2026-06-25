# EXPLAINER — knowledge-graph-rag: retrieval that walks a graph

## What I implemented from scratch

- **Entity + relation extractor** (`core/extract.py`)
- **In-memory knowledge graph** with BFS/Dijkstra/subgraph (`core/graph.py`)
- **Graph-augmented retrieval** over a from-scratch BM25-lite scorer (`core/retrieval.py`)

No spaCy, no networkx, no rank_bm25. Below is how each piece works and why the
combination beats plain lexical RAG on multi-hop questions.

## Extraction: entities

Three reinforcing signals find entities without a trained model:

1. **Capitalization runs.** Consecutive capitalized tokens merge into one span
   ("Gamma Labs", "Jane Smith"). Lowercase *connectors* (`of`, `and`, `the`,
   `&`) are allowed inside a span only when a capitalized token follows, so
   "Bank of America" stays whole but "Acme in 2022" does not absorb "in".
2. **Org suffixes** (`Inc`, `Corp`, `Ltd`, `Labs`, ...) greedily extend a span,
   so "Beta Inc" is one entity, not two.
3. **A tiny gazetteer** of role words (`CEO`, `Founder`, ...) seeds entities the
   capitalization heuristic alone would treat as common nouns.

A sentence-initial capitalized **stop word** ("The", "After") is dropped so the
mandatory leading capital of English doesn't manufacture junk entities.

## Extraction: relations

Two complementary mechanisms produce `(head, relation, tail)` triples:

- **Verb templates.** A table of cue patterns maps surface verbs to canonical
  relations: `acquired|bought → acquired`, `founded|established → founded`,
  `merged with → merged_with`, `is based in → based_in`, etc. For each match we
  split the sentence at the cue, take the **nearest entity before** it as the
  head and the **first entity after** it as the tail.
- **A role pattern** captures "X is the CEO of Y" → `(X, is_ceo_of, Y)`.
- **Co-occurrence.** Every distinct entity pair inside the same sentence is
  linked with a `co_occurs` edge whose **weight is the running count** of how
  often that pair appears together — an untyped association that boosts recall
  where no verb template fires.

## The graph (`graph.py`)

Storage is a directed weighted adjacency map: `node -> [(neighbor, relation,
weight)]`. Adding a relation that already exists **accumulates weight** rather
than duplicating the edge, so repeated co-occurrences strengthen a link.

- `neighbors(node, undirected=?)` lists outgoing edges, plus incoming ones in
  undirected mode (Beta is acquired *by* Acme, but the two are still connected
  for recall).
- `shortest_path` is **Dijkstra** over a mirrored undirected adjacency, with
  edge cost `1/weight` so stronger relations are cheaper to traverse. With
  uniform weights this reduces to BFS hop-count. It reconstructs the node path
  from a predecessor map and returns `None` when the target is unreachable.
- `subgraph_for_query` matches query terms to **seed nodes** (multi-word names
  by substring, single tokens on word boundaries) and runs a **bounded BFS** to
  depth `k` from all seeds at once, returning the connected neighborhood.

## How the graph augments retrieval

The baseline is a from-scratch **BM25-lite** scorer: per-term IDF with `+0.5`
smoothing, term-frequency saturation via `k1`, and document-length
normalization via `b`. It ranks passages on overlap with the query's surface
terms — and *only* those terms.

`graph_augmented_retrieve` adds one step before scoring:

1. Match the query to seed entities in the graph.
2. Walk the subgraph out to `hops` hops, collecting reachable entities.
3. Append those entity surface forms as **extra query terms**.
4. Re-run BM25 with the enriched term set.

This is why it improves multi-hop recall: a question can name only `Acme`, yet
the answer lives in a passage about `Gamma` that shares no words with the
question. Lexical retrieval scores that passage at zero. But the graph knows
`Acme → acquired → Beta → owns → Gamma`, so expansion injects `Gamma` into the
query and the answer passage jumps to the top.

## Worked example

Corpus:

```
P0: Acme is a large holding company headquartered downtown.
P1: Gamma manufactures industrial robotics for warehouses.   <- the answer
P2: The cafeteria introduced a seasonal menu in spring.
```

Graph (from a separate ingest): `Acme --acquired--> Beta --owns--> Gamma`.

Question: **"What does Acme ultimately produce?"**

- **Baseline:** query terms `{what, does, acme, ultimately, produce}` hit only
  P0 ("acme"). P1 scores 0 — **missed**.
- **Graph-augmented (hops=2):** seed `Acme`; BFS reaches `Beta`, `Gamma`. Terms
  become `{... acme ..., beta, gamma}`. Now P1 contains "gamma" → it ranks
  **first**. The answer is recovered purely through graph structure.

`tests/test_core.py::test_graph_augmentation_beats_baseline` asserts exactly
this: P1 absent from the baseline result, present and top-ranked with expansion.

## Limitations

- Pattern/gazetteer extraction covers common templates, not arbitrary phrasing.
- `co_occurs` edges are untyped associations — great for recall, noisier at high
  hop counts (cap `hops` to keep expansion focused).
- BM25-lite is lexical; a dense retriever would compose on top of the same graph
  layer unchanged.
