# 🕸️ knowledge-graph-rag — entity-graph-augmented RAG

## What I implemented from scratch

- **Entity + relation extractor** — proper-noun detection (capitalization runs + connectors + org suffixes), a tiny gazetteer, and regex; relation extraction via verb templates (`X acquired Y`, `X founded Y`, `X is the CEO of Y`) **plus** sentence-window co-occurrence with counts — `core/extract.py`
- **In-memory knowledge graph** — weighted adjacency map, `add_entity`/`add_relation`, directed + undirected `neighbors`, Dijkstra `shortest_path`, and `subgraph_for_query` (bounded BFS from query-matched seeds). No networkx — `core/graph.py`
- **Graph-augmented retrieval** — a from-scratch BM25-lite lexical retriever, plus graph expansion that matches query terms to entities, walks the subgraph, and re-retrieves with the new entity terms appended — `core/retrieval.py`

No spaCy, no networkx, no rank_bm25, no sklearn. The graph, the extractor, and the scorer are all hand-written.

## Why it's here

Naive RAG retrieves on surface terms, so it misses passages that answer a question *indirectly* — the relevant evidence lives under an entity the question never names. This project builds a knowledge graph from the corpus and uses it to **expand the query along relations** ("Acme → acquired → Beta → owns → Gamma"), recovering multi-hop evidence a lexical retriever alone would never surface.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000   # http://localhost:8000
```

Ingest a corpus, then ask:

```bash
# 1. ingest documents
curl -s -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{
  "documents": [
    "Acme Corp acquired Beta Inc in 2022 for 12 million dollars.",
    "Beta Inc owns Gamma Labs, a robotics subsidiary.",
    "Gamma Labs manufactures autonomous warehouse robots."
  ]
}'

# 2. ask a multi-hop question (graph expansion on by default)
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' -d '{
  "question": "What does Acme ultimately produce?", "use_graph": true, "k": 3
}'

# 3. shortest path between two entities
curl -s "localhost:8000/graph/path?src=Acme%20Corp&dst=Gamma%20Labs"
```

## API

| Method | Path           | Body / Query                                   | Returns                                              |
|--------|----------------|------------------------------------------------|------------------------------------------------------|
| GET    | `/`            | —                                              | service banner + endpoint list                       |
| GET    | `/health`      | —                                              | `{status, passages, nodes, edges}`                   |
| POST   | `/ingest`      | `{text?: str, documents?: [str]}`              | `{passages_added, total_passages, entities, edges}`  |
| POST   | `/ask`         | `{question, use_graph?, k?, hops?}`            | `{answer, passages, entities_expanded, path?, used_graph}` |
| GET    | `/graph`       | —                                              | `{num_nodes, num_edges, nodes, edges}`               |
| GET    | `/graph/path`  | `?src=&dst=`                                   | `{src, dst, path, found}`                            |

All requests are Pydantic-validated; bad input returns `422`, unknown entities `404`, empty state `400`.

## Verify

```bash
pytest -q
```

The suite proves the extractor pulls `(Acme, acquired, Beta)`, the graph finds a known multi-hop shortest path and connected subgraph, and — the key test, `test_graph_augmentation_beats_baseline` — that on a crafted corpus where the answer passage shares **no** terms with the question, graph expansion retrieves and top-ranks it while the lexical baseline misses it entirely.

## Limitations

- Extraction is pattern/gazetteer based — it handles the common org/role/acquisition templates but won't match arbitrary phrasing the way a trained NER/RE model would.
- Co-occurrence edges are undirected associations, not typed relations; they help recall but add noise at high hop counts.
- The retriever is lexical (BM25-lite); a dense embedding retriever is a drop-in upgrade for paraphrase matching, and the graph layer would compose on top unchanged.
