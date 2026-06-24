"""Proofs for the agentic RAG planner, verifier, and loop."""

from core.planner import plan_query
from core.verifier import verify_answer, groundedness
from core.loop import agentic_answer


def test_planner_splits_conjunctions():
    subs = plan_query("What is the revenue and what is the profit margin?")
    assert len(subs) >= 2


def test_planner_handles_comparison():
    subs = plan_query("How does 2023 revenue compare to 2022 revenue?")
    assert len(subs) == 2


def test_planner_single_question_passthrough():
    subs = plan_query("What is the capital of France?")
    assert subs == ["What is the capital of France?"]


def test_verifier_flags_unsupported_claim():
    sources = ["The company reported revenue of 10 million dollars in 2023."]
    answer = "Revenue was 10 million dollars in 2023. The CEO resigned in protest."
    checks = verify_answer(answer, sources)
    # first claim is supported, second is not in the sources
    supported = {c.claim: c.supported for c in checks}
    assert any("10 million" in c and v for c, v in supported.items())
    assert any("resigned" in c and not v for c, v in supported.items())


def test_groundedness_score():
    sources = ["Solar panels convert sunlight into electricity."]
    answer = "Solar panels convert sunlight into electricity."
    checks = verify_answer(answer, sources)
    assert groundedness(checks) == 1.0


def test_agentic_loop_runs_and_grounds():
    # a trivial keyword retriever over a tiny corpus
    corpus = [
        "Acme Corp revenue grew to 50 million in 2023.",
        "Acme Corp acquired Beta Inc in 2022.",
        "The weather was sunny.",
    ]

    def retriever(query, k):
        ql = set(query.lower().split())
        scored = sorted(corpus, key=lambda d: -len(ql & set(d.lower().split())))
        return scored[:k]

    trace = agentic_answer("What was Acme revenue and who did they acquire?", retriever)
    assert trace.sub_questions  # it planned
    assert trace.passages       # it retrieved
    assert trace.answer         # it synthesized
    assert 0.0 <= trace.groundedness <= 1.0
    assert trace.rounds >= 1
