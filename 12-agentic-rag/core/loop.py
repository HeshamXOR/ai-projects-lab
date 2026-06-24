"""The agentic RAG loop: plan -> retrieve -> synthesize -> verify -> refine.

This orchestrates the from-scratch planner and verifier into a multi-step
reasoning loop, the way an agentic RAG system works:

  1. PLAN     — decompose the question into sub-questions.
  2. RETRIEVE — gather passages for each sub-question (iteratively).
  3. SYNTH    — compose an answer from the retrieved context.
  4. VERIFY   — check each claim is grounded (citation verifier).
  5. CRITIQUE — if groundedness is low, retrieve more and retry (up to N rounds).

The retriever and the synthesizer are injected (dependency injection), so the
loop is testable with a trivial deterministic retriever and works with a real
embedding retriever + LLM in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from .planner import plan_query
from .verifier import verify_answer, groundedness, ClaimCheck

# A retriever maps (query, k) -> list of passage strings.
Retriever = Callable[[str, int], List[str]]
# A synthesizer maps (question, context_passages) -> answer string.
Synthesizer = Callable[[str, List[str]], str]


@dataclass
class Trace:
    sub_questions: List[str] = field(default_factory=list)
    rounds: int = 0
    passages: List[str] = field(default_factory=list)
    answer: str = ""
    checks: List[ClaimCheck] = field(default_factory=list)
    groundedness: float = 0.0


def default_synthesizer(question: str, context: List[str]) -> str:
    """A no-LLM fallback: stitch the most relevant retrieved sentences.

    Keeps the demo working without an API key; a real synthesizer is injected
    when an LLM is configured.
    """
    import re

    sents = []
    for c in context:
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", c) if len(s.strip()) > 20]
    return " ".join(sents[:4]) if sents else "No relevant information found."


def agentic_answer(
    question: str,
    retriever: Retriever,
    synthesizer: Synthesizer = default_synthesizer,
    llm=None,
    k: int = 3,
    max_rounds: int = 3,
    target_groundedness: float = 0.7,
) -> Trace:
    """Run the full plan→retrieve→synthesize→verify→refine loop."""
    trace = Trace()
    trace.sub_questions = plan_query(question, llm=llm)

    collected: List[str] = []
    seen = set()
    for round_i in range(1, max_rounds + 1):
        trace.rounds = round_i
        # retrieve for each sub-question (widen k each round to find more)
        for sq in trace.sub_questions:
            for p in retriever(sq, k + round_i - 1):
                if p not in seen:
                    seen.add(p)
                    collected.append(p)
        trace.passages = collected

        # synthesize an answer from everything gathered so far
        trace.answer = synthesizer(question, collected)

        # verify groundedness
        trace.checks = verify_answer(trace.answer, collected)
        trace.groundedness = groundedness(trace.checks)

        # self-critique: good enough? stop. otherwise loop and gather more.
        if trace.groundedness >= target_groundedness:
            break
    return trace
