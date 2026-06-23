# 🔎 Semantic Code Search

Index a folder of source code and search functions/classes by **natural language** — "where do we validate the auth token?" — instead of grepping for exact strings.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What it does

- **Parses code into units** — Python functions/classes via the `ast` module; a brace/keyword heuristic for JS/TS/Java/Go/Rust/C/C++/C#/Ruby.
- **Embeds each block** — encodes `name + code` with `all-MiniLM-L6-v2`.
- **Semantic search** — ranks blocks by cosine similarity to your query and shows the code with file + line number.

## Why it's real

Developers constantly search code by intent, not exact names. This is the core of tools like Sourcegraph's semantic search and IDE "find by description" features — useful on any real codebase.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

On startup it indexes its **own folder** so the preview works immediately — then point it at any path (e.g. `..` to index the whole lab). CPU-only.

## How it works (files)

- `codesearch.py` — `extract_blocks()` (AST for Python, heuristic otherwise), `CodeIndex.index_folder / search`.
- `app.py` — Gradio UI: index a folder → search → ranked code results.

## Extend it

- Add tree-sitter for accurate multi-language parsing.
- Persist the index to disk (FAISS) for large repos.
- Add a code-specific embedding model (e.g. a CodeBERT variant) for better ranking.
