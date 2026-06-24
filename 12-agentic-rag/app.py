"""agentic-rag — multi-step reasoning RAG with self-verification.

Paste a small corpus and ask a (possibly multi-hop) question. The app shows the
agent's trace: how it decomposed the question, what it retrieved, the synthesized
answer, and a per-claim groundedness check — all driven by from-scratch logic in
core/ (planner, verifier, loop). Optionally uses an embedding retriever + LLM if
available; otherwise a lexical retriever + extractive synthesizer keep it
running with no API key.
"""

from __future__ import annotations

import os
import re

import gradio as gr

from core.loop import agentic_answer, default_synthesizer

SAMPLE = (
    "Acme Corp reported revenue of 50 million dollars in 2023, up from 38 million in 2022. "
    "In 2022, Acme Corp acquired Beta Inc, a logistics startup, for 12 million dollars. "
    "The acquisition expanded Acme's distribution network across three new regions. "
    "Acme's CEO said the company plans to reach 80 million in revenue by 2025. "
    "Beta Inc was founded in 2018 and had 40 employees at the time of acquisition. "
    "Unrelated: the office cafeteria introduced a new menu in spring."
)


def _split_corpus(text: str):
    # one passage per sentence (simple, transparent chunking)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 15]


def _make_retriever(passages):
    def retriever(query, k):
        ql = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = sorted(
            passages, key=lambda d: -len(ql & set(re.findall(r"[a-z0-9]+", d.lower())))
        )
        return scored[:k]
    return retriever


def _llm():
    """Optional LLM synthesizer if a key is set; else None."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None

    def call(prompt):
        import httpx

        base = os.environ.get("OPENFORGE_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENFORGE_MODEL", "gpt-4o-mini")
        r = httpx.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    return call


def on_ask(corpus, question):
    passages = _split_corpus(corpus)
    if not passages or not question.strip():
        return "Provide a corpus and a question."
    retriever = _make_retriever(passages)
    llm = _llm()

    if llm:
        def synth(q, ctx):
            ctx_block = "\n".join(f"- {c}" for c in ctx)
            return llm(f"Answer using ONLY these facts:\n{ctx_block}\n\nQuestion: {q}\nAnswer:")
    else:
        synth = default_synthesizer

    trace = agentic_answer(question, retriever, synthesizer=synth, llm=llm)

    plan = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(trace.sub_questions))
    claims = "\n".join(
        f"- {'✅' if c.supported else '⚠️'} ({c.score:.2f}) {c.claim}" for c in trace.checks
    )
    return (
        f"### 🧭 Plan ({trace.rounds} retrieval round(s))\n{plan}\n\n"
        f"### 💬 Answer\n{trace.answer}\n\n"
        f"### 🔎 Groundedness: **{trace.groundedness:.0%}** of claims supported\n{claims}\n\n"
        f"_{len(trace.passages)} passages retrieved. ⚠️ = claim not well-supported by the corpus._"
    )


with gr.Blocks(title="agentic-rag", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤖 agentic-rag — multi-step RAG that checks its own work\n"
        "Ask a multi-hop question over a corpus. The agent **decomposes** it, "
        "**retrieves iteratively**, **synthesizes** an answer, and **verifies** "
        "that each claim is grounded in the sources — flagging anything that "
        "isn't. Planner, verifier, and loop are all from scratch (`core/`)."
    )
    corpus = gr.Textbox(label="Corpus", lines=8, value=SAMPLE)
    question = gr.Textbox(label="Question", value="What was Acme's revenue and who did they acquire?")
    btn = gr.Button("Ask", variant="primary")
    out = gr.Markdown()
    btn.click(on_ask, [corpus, question], out)
    gr.Examples(
        examples=[
            "What was Acme's revenue and who did they acquire?",
            "How did 2023 revenue compare to 2022?",
            "When was Beta Inc founded and how many employees did it have?",
        ],
        inputs=question,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
