# EXPLAINER — Code Search: a real index + ranking + call graph

## What I implemented from scratch

- **Inverted index** with positional postings — boolean AND and phrase queries (`core/inverted_index.py`).
- **BM25F** — field-weighted BM25 that ranks code by *where* a term matches (name > signature > comments > body) (`core/bm25f.py`).
- **Call graph** — who-calls-whom from the Python AST, for "related functions" (`core/callgraph.py`).

The embedding model (existing) provides semantic recall; these add lexical precision and structure.

## Why each piece

**Inverted index**: the data structure behind every search engine. Instead of scanning files, map term → (doc, positions) and intersect short postings lists. Positions let us answer *phrase* queries ("read the file" as an adjacent sequence), not just "contains these words."

**BM25F**: plain BM25 treats a function as a flat bag of words, so a query term in a comment counts as much as one in the function name. Code is structured — BM25F weights fields before the term-frequency saturation, so a match in `read_file`'s *name* outranks an incidental `read` in someone's comment. The tokenizer also splits `camelCase`/`snake_case`, so `readFile` matches `read` and `file`.

**Call graph**: parse the AST, and for each function record the functions it calls. Invert that for "what calls this?" One hop in either direction gives "related functions" — the structural navigation IDEs provide.

## Proof it works

`tests/test_core.py`:
- Inverted index does boolean-AND and correctly restricts phrase matches by position.
- BM25F ranks the function *named* `read_file` above one that merely says "read" several times in its body — proving field weighting works.
- camelCase/snake_case tokenization splits identifiers.
- The call graph resolves callees, callers, and related functions on a known snippet.

## Limitations

- Call graph is Python-only (AST); other languages use the heuristic chunker from the original project.
- Name resolution is by simple name (not scope-aware) — good enough for navigation, not a full type-resolver.
