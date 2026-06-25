"""Tests for the policy engine, scoring, explanations, and full pipeline."""

import pytest

from core.policy import PolicyEngine
from core.rules import Category, default_ruleset
from core.scoring import Decision, score_text


# --------------------------- Policy engine --------------------------------- #
def test_engine_collects_expected_hits(policy):
    hits = policy.evaluate("you are a worthless idiot and i hate you")
    rule_ids = {h.rule_id for h in hits}
    assert "tox.insults" in rule_ids
    assert "tox.hate" in rule_ids


def test_engine_groups_by_category(policy):
    grouped = policy.hits_by_category("email me bob@x.com you idiot")
    assert grouped[Category.PII]
    assert grouped[Category.TOXICITY]


def test_engine_clean_text_no_hits(policy):
    assert policy.evaluate("thank you for your kind help today") == []


def test_engine_rejects_duplicate_ids():
    rules = default_ruleset()
    rules.append(rules[0])  # duplicate id
    with pytest.raises(ValueError):
        PolicyEngine(rules)


def test_hits_sorted_by_position(policy):
    hits = policy.evaluate("idiot ... bob@x.com")
    positions = [h.start for h in hits]
    assert positions == sorted(positions)


# --------------------------- Scoring --------------------------------------- #
def test_clean_allows():
    res = score_text([], toxic_probability=0.05)
    assert res.decision == Decision.ALLOW
    assert res.overall < 0.35


def test_high_severity_blocks():
    engine = PolicyEngine()
    hits = engine.evaluate("kill yourself")
    res = score_text(hits, toxic_probability=0.9)
    assert res.decision == Decision.BLOCK


def test_severity_ordering():
    """A more severe input must score at least as high as a milder one."""
    engine = PolicyEngine()
    mild = score_text(engine.evaluate("you fool"), toxic_probability=0.5)
    severe = score_text(engine.evaluate("kill yourself you idiot"),
                        toxic_probability=0.95)
    assert severe.overall >= mild.overall


def test_classifier_alone_can_flag():
    # No rules fire, but the classifier is confident -> at least a flag.
    res = score_text([], toxic_probability=0.95)
    assert res.decision in (Decision.FLAG, Decision.BLOCK)


def test_score_threshold_validation():
    with pytest.raises(ValueError):
        score_text([], 0.5, flag_threshold=0.8, block_threshold=0.5)


def test_category_scores_bounded():
    engine = PolicyEngine()
    res = score_text(engine.evaluate("kill yourself idiot"), 1.0)
    for v in res.category_scores.values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= res.overall <= 1.0


# --------------------------- Full pipeline --------------------------------- #
def test_pipeline_allows_clean_text(pipeline):
    result = pipeline.moderate("thank you so much for the wonderful help")
    assert result.decision == Decision.ALLOW
    assert result.hits == []


def test_pipeline_blocks_toxic(pipeline):
    result = pipeline.moderate("you are a worthless stupid idiot")
    assert result.decision == Decision.BLOCK


def test_pipeline_blocks_self_harm(pipeline):
    result = pipeline.moderate("kill yourself you loser")
    assert result.decision == Decision.BLOCK
    cats = {h.category for h in result.hits}
    assert Category.SELF_HARM in cats


def test_pipeline_flags_pii(pipeline):
    result = pipeline.moderate("reach me at alice@example.com")
    assert result.decision in (Decision.FLAG, Decision.BLOCK)
    assert any(h.rule_id == "pii.email" for h in result.hits)


def test_pipeline_blocks_card(pipeline):
    result = pipeline.moderate("here is my card 4242 4242 4242 4242")
    assert result.decision == Decision.BLOCK
    assert any(h.rule_id == "pii.credit_card" for h in result.hits)


def test_pipeline_explanation_structure(pipeline):
    result = pipeline.moderate("you stupid idiot")
    exp = result.explanation.to_dict()
    assert "summary" in exp
    assert "triggered_rules" in exp
    assert "category_breakdown" in exp
    assert exp["triggered_rules"]
    assert exp["classifier"]["contributed_to"] == "toxicity"


def test_pipeline_batch(pipeline):
    results = pipeline.moderate_batch(["thanks a lot", "you idiot"])
    assert len(results) == 2
    assert results[0].decision == Decision.ALLOW
    assert results[1].decision in (Decision.FLAG, Decision.BLOCK)


def test_pipeline_stream_independent(pipeline):
    chunks = ["hello there", "you worthless idiot", "thanks again"]
    verdicts = list(pipeline.moderate_stream(chunks))
    assert len(verdicts) == 3
    assert verdicts[0].decision == Decision.ALLOW
    assert verdicts[1].decision in (Decision.FLAG, Decision.BLOCK)


def test_pipeline_stream_cumulative(pipeline):
    chunks = ["i will", "hurt you"]
    verdicts = list(pipeline.moderate_stream(chunks, cumulative=True))
    # The threat only completes once both chunks are combined.
    assert verdicts[-1].decision in (Decision.FLAG, Decision.BLOCK)


def test_pipeline_rejects_non_string(pipeline):
    with pytest.raises(TypeError):
        pipeline.moderate(123)
