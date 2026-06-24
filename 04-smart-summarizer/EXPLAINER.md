# EXPLAINER — Smart Summarizer: the algorithm, not just the model

## What I implemented from scratch

- **TextRank** extractive summarization — PageRank on a sentence-similarity graph, solved by power iteration (`core/textrank.py`).
- **ROUGE** evaluation (ROUGE-1, ROUGE-2, ROUGE-L with LCS) — the standard summarization metric (`core/rouge.py`).

The transformer summarizer (DistilBART) stays as the abstractive option; now it sits *next to* a from-scratch extractive method you can compare against.

## How TextRank works

1. Split text into sentences; each becomes a graph node.
2. Weight every edge by sentence similarity (word overlap normalized by length).
3. Run **power iteration**: `r = (1−d)/n + d·Mᵀr` until it converges. This is the PageRank recurrence — a sentence is important if it's similar to other important sentences. The converged vector `r` is the eigenvector of the transition matrix.
4. Take the top-k sentences (in original order) as the summary.

No model, no training — pure graph centrality. It runs instantly on CPU and is fully interpretable.

## How ROUGE works

- **ROUGE-N**: precision/recall/F1 over n-gram overlap between candidate and reference.
- **ROUGE-L**: based on the **longest common subsequence** (classic DP), rewarding summaries that keep the reference's word order, not just its words.

## Proof it works

`tests/test_core.py`:
- TextRank picks the document's central topic and rejects off-topic sentences; preserves original order; returns everything for very short inputs.
- ROUGE scores identical text as 1.0, disjoint text as 0.0, and ROUGE-L correctly rewards in-order overlap over shuffled overlap.

## Limitations

- Lexical-overlap similarity (not embeddings) keeps TextRank dependency-free; an embedding-based edge weight would improve it.
- Extractive summaries quote the source verbatim (by design); the abstractive model paraphrases. The app shows both so the trade-off is visible.
