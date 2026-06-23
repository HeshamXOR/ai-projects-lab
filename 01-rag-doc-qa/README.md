# 📄 RAG Document Q&A

Upload a PDF, ask questions, and get answers **grounded in the document** with page citations. A classic Retrieval-Augmented Generation pipeline with a clean Gradio UI.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What it does

1. **Chunk** — splits the PDF into overlapping text chunks, tagged by page.
2. **Embed** — encodes chunks with `all-MiniLM-L6-v2` (Sentence-Transformers).
3. **Retrieve** — for each question, finds the most semantically similar chunks (cosine similarity).
4. **Answer** — stitches the most relevant sentences into a cited answer (extractive, no API key), or generates one if `OPENAI_API_KEY` is set.

## Why it's real

Document Q&A is one of the most common applied-AI tasks — support docs, contracts, research papers, policies. This is the core pattern behind "chat with your PDF" products, built from scratch so you understand every step.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ a public gradio.live link)
```

Optional generated answers:
```bash
export OPENAI_API_KEY=sk-...        # any OpenAI-compatible key
# export OPENFORGE_BASE_URL=...     # to point at a different endpoint
python app.py
```

## How it works (files)

- `rag.py` — the engine: `extract_pdf_chunks`, `RagEngine.index_pdf / retrieve / answer`.
- `app.py` — the Gradio UI: upload → index → chat.

## Notes & limitations

- Works on **CPU**; embeddings are fast even without a GPU.
- Scanned/image-only PDFs have no extractable text — add OCR (e.g. `pytesseract`) to handle those.
- Extractive mode quotes the document directly; generative mode (with a key) reads more fluently.
