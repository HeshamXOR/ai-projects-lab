"""Proofs for TextRank and ROUGE."""

from core.textrank import textrank, summarize_extractive, split_sentences
from core.rouge import rouge_n, rouge_l, rouge_report


DOC = (
    "Solar power is a renewable energy source. "
    "Solar panels convert sunlight into electricity using photovoltaic cells. "
    "The weather today is sunny and warm. "
    "Photovoltaic technology has improved dramatically and costs have fallen. "
    "Many homeowners now install solar panels to reduce their electricity bills. "
    "My favorite color is blue."
)


def test_textrank_picks_central_sentences():
    # the central topic is solar/photovoltaic; off-topic sentences (weather,
    # favorite color) should NOT dominate the top summary
    summary = textrank(DOC, top_k=3)
    joined = " ".join(summary).lower()
    assert "solar" in joined or "photovoltaic" in joined
    assert "favorite color is blue" not in joined


def test_textrank_preserves_order():
    summary = textrank(DOC, top_k=3)
    sentences = split_sentences(DOC)
    positions = [sentences.index(s) for s in summary]
    assert positions == sorted(positions)


def test_textrank_short_doc_returns_all():
    short = "One sentence only."
    assert textrank(short, top_k=3) == ["One sentence only."]


def test_rouge_identical_is_perfect():
    r = rouge_n("the cat sat", "the cat sat", 1)
    assert abs(r["f1"] - 1.0) < 1e-9


def test_rouge_no_overlap_is_zero():
    r = rouge_n("alpha beta", "gamma delta", 1)
    assert r["f1"] == 0.0


def test_rouge_l_rewards_order():
    # same words, but candidate preserves reference order better
    ref = "the quick brown fox"
    good = "the quick brown fox"
    assert rouge_l(good, ref)["f1"] == 1.0
    partial = rouge_l("quick the fox brown", ref)["f1"]
    assert 0 < partial < 1.0


def test_rouge_report_keys():
    rep = rouge_report("the cat", "the cat sat")
    assert set(rep) == {"rouge-1", "rouge-2", "rouge-l"}
