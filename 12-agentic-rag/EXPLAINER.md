# EXPLAINER — agentic-rag: RAG that reasons and self-checks

## What I implemented from scratch

- **Query planner** — decomposes a question into retrievable sub-questions (`core/planner.py`).
- **Citation verifier** — grounds each answer claim in a source, scores groundedness (`core/verifier.py`).
- **Agentic loop** — plan → retrieve → synthesize → verify → refine (`core/loop.py`).

The embedding retriever and LLM synthesizer are *optional injected components*; the orchestration is mine.

## Why naive RAG isn't enough

Standard RAG does one retrieval and feeds it to the model. Two failure modes:
1. **Multi-hop questions** ("revenue change *after* the acquisition") need multiple, different retrievals.
2. **Hallucination** — the model can assert things the sources don't support, and naive RAG never checks.

agentic-rag addresses both.

## The loop (`loop.py`)

1. **Plan** — `plan_query` breaks the question into sub-questions (splitting conjunctions and comparatives; an LLM hook upgrades this).
2. **Retrieve** — for each sub-question, pull top-k passages; widen k each round to gather more on retries.
3. **Synthesize** — compose an answer from the gathered context (extractive fallback, or an injected LLM).
4. **Verify** — `verify_answer` splits the answer into claims and grounds each against the sources.
5. **Critique & refine** — if groundedness is below target, loop again and retrieve more. This is the "self-checking" behavior that makes the system trustworthy.

Dependency injection (retriever + synthesizer as parameters) is what makes the loop unit-testable with a trivial retriever and production-ready with real components.

## The verifier (`verifier.py`)

For each answer sentence, find the source passage with the highest content-word **Jaccard overlap**. Above a threshold → supported; below → flagged ⚠️. `groundedness` = fraction supported, a single trust score. The test confirms it marks an in-source claim supported and an invented one ("the CEO resigned") unsupported.

## Proof it works

`tests/test_core.py`:
- Planner splits conjunctions and comparison questions, passes single questions through unchanged.
- Verifier flags an unsupported claim while accepting a supported one; perfect grounding scores 1.0.
- The full loop plans, retrieves, synthesizes, and returns a groundedness score on a tiny corpus.

## Limitations

- Lexical (Jaccard) grounding misses paraphrased support; an NLI or embedding-similarity verifier is the upgrade path.
- Rule-based planning is deterministic but simpler than LLM decomposition (which the hook enables).
