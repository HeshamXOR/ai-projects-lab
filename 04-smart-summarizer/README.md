# 📝 Smart Summarizer

Paste a long article, report, or meeting transcript → get a concise **summary** plus extracted **action items**. Handles long inputs by chunking and map-reduce summarization.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **TextRank** extractive summarization (PageRank via power iteration on a sentence graph) — `core/textrank.py`
- **ROUGE** evaluation (ROUGE-1/2/L) — `core/rouge.py`

The transformer summarizer is the abstractive option; the app compares it against the from-scratch extractive method and scores them with from-scratch ROUGE. See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## What it does

- **Chunked summarization** — splits long text into word-bounded chunks, summarizes each, then summarizes the combination (map-reduce), so arbitrarily long inputs work despite the model's token limit.
- **Action-item extraction** — heuristically pulls out commitments and to-dos ("will…", "need to…", "by next Friday", "action item…") — perfect for meeting notes.
- **Compression stats** — shows how much the text was condensed.

## Why it's real

Summarization is one of the highest-value NLP tasks in practice — meeting notes, news digests, research triage, support-ticket condensation. The action-item extraction makes it genuinely useful for teams.

## Run it

```bash
pip install -r requirements.txt
python app.py            # http://localhost:7860  (+ public gradio.live link)
```

Pre-filled with a sample meeting transcript so the preview works on first click. GPU (L4) makes long inputs faster; CPU works for demos.

## How it works (files)

- `summarizer.py` — `summarize()` (chunk → summarize → reduce), `extract_action_items()`, GPU/CPU auto-detect. Model: `distilbart-cnn-12-6`.
- `app.py` — the Gradio UI.

## Extend it

- Add per-section summaries for structured documents.
- Swap in a larger summarization model for higher quality on a bigger GPU.
- Feed in transcripts from an ASR model for end-to-end meeting summarization.
