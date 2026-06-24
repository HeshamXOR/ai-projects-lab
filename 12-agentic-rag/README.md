# 🤖 agentic-rag — multi-step RAG that checks its own work

A retrieval-augmented Q&A system that **decomposes** a question, **retrieves iteratively**, **synthesizes** an answer, and **verifies its own claims** against the sources — flagging anything unsupported. The advanced-RAG counterpart to project 01.

![preview](preview.gif)
<!-- Record a short clip on Lightning and save it as preview.gif here. -->

## What I implemented from scratch

- **Query planner** — decomposes multi-hop questions into sub-questions — `core/planner.py`
- **Citation verifier** — checks each answer sentence is grounded in a retrieved passage, with a groundedness score — `core/verifier.py`
- **The agentic loop** — plan → retrieve → synthesize → verify → refine (retry if under-grounded) — `core/loop.py`

The retriever and synthesizer are injected, so it runs with a lexical retriever + extractive synthesizer (no API key) and upgrades to embeddings + an LLM when available. See [EXPLAINER.md](EXPLAINER.md); verify with `pytest`.

## Why it's here

Naive RAG retrieves once and hopes. This shows the *reasoning* layer that makes RAG reliable: breaking down hard questions, gathering evidence over multiple rounds, and — crucially — **not trusting the model's output blindly** but checking every claim against the sources.

## Run it

```bash
pip install -r requirements.txt
python app.py        # http://localhost:7860 (+ public gradio.live link)
```

Paste a corpus, ask a multi-hop question, and read the full trace: the plan, the answer, and a per-claim ✅/⚠️ groundedness check. Optional: set `OPENAI_API_KEY` for LLM synthesis.

## Verify

```bash
pytest -q   # planner decomposition, verifier flags unsupported claims, full loop runs
```

## Limitations

- Verifier uses lexical overlap (Jaccard); an NLI/embedding check would catch paraphrased support better (drop-in upgrade).
- The planner is rule-based by default (deterministic + testable); the LLM hook improves decomposition when a key is set.
