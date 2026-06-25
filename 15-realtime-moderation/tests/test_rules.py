"""Tests for the rule DSL: Luhn, PII detectors, heuristics, keyword/regex."""

import pytest

from core.rules import (
    Category,
    Rule,
    RuleType,
    default_ruleset,
    luhn_check,
)


# --------------------------- Luhn checksum --------------------------------- #
def test_luhn_valid_known_cards():
    # Well-known Luhn-valid test card numbers (16 digits each).
    assert luhn_check("4242424242424242")  # Visa test card
    assert luhn_check("4111111111111111")  # Visa test card
    assert luhn_check("5500005555555559")  # Mastercard test card


def test_luhn_strips_separators():
    # The same number with spaces/dashes must still validate.
    assert luhn_check("4242 4242 4242 4242")
    assert luhn_check("4242-4242-4242-4242")


def test_luhn_invalid():
    assert not luhn_check("4242424242424241")  # last digit off by one
    assert not luhn_check("1234567890123456")
    assert not luhn_check("0000000000000001")


def test_luhn_too_short():
    assert not luhn_check("4242")
    assert not luhn_check("")


# --------------------------- PII detectors --------------------------------- #
def _rule(rule_id):
    return next(r for r in default_ruleset() if r.id == rule_id)


def test_email_detector():
    rule = _rule("pii.email")
    spans = rule.apply("contact me at jane.doe+test@example.co.uk please")
    assert len(spans) == 1
    assert "jane.doe+test@example.co.uk" in spans[0][2]


def test_phone_detector():
    rule = _rule("pii.phone")
    spans = rule.apply("call (555) 123-4567 today")
    assert len(spans) >= 1


def test_ssn_detector():
    rule = _rule("pii.ssn")
    spans = rule.apply("my ssn is 123-45-6789 ok")
    assert len(spans) == 1
    assert spans[0][2] == "123-45-6789"


def test_credit_card_detector_only_valid():
    rule = _rule("pii.credit_card")
    valid = rule.apply("card 4242 4242 4242 4242 here")
    assert len(valid) == 1
    invalid = rule.apply("number 1234 5678 9012 3456 here")
    assert len(invalid) == 0


# --------------------------- Heuristics ------------------------------------ #
def test_excessive_caps():
    rule = _rule("spam.caps")
    assert rule.apply("THIS IS ALL SHOUTING NONSENSE")
    assert not rule.apply("This is normal sentence casing here")


def test_repeated_chars():
    rule = _rule("spam.repeated_chars")
    assert rule.apply("soooooo good")
    assert rule.apply("wow!!!!!!")
    assert not rule.apply("normal text")


def test_repeated_tokens():
    rule = _rule("spam.repeated_tokens")
    assert rule.apply("buy buy buy buy buy")
    assert not rule.apply("buy one get one")


def test_url_flood():
    rule = _rule("spam.url_flood")
    text = "see http://a.com and http://b.com plus http://c.com now"
    assert rule.apply(text)
    assert not rule.apply("see http://a.com only")


# --------------------------- Keyword / regex ------------------------------- #
def test_keyword_word_boundary():
    rule = _rule("tox.insults")
    # "idiot" should match, but not as a substring of "idiotic"? word boundary
    # makes "idiot" match the standalone token.
    assert rule.apply("you are an idiot")
    # No insult words here.
    assert not rule.apply("the assistant was helpful")


def test_regex_threat():
    rule = _rule("tox.threat")
    assert rule.apply("i will hurt you badly")
    assert not rule.apply("i will help you today")


def test_rule_validation_rejects_bad_severity():
    with pytest.raises(ValueError):
        Rule(id="x", category=Category.SPAM, rule_type=RuleType.KEYWORD,
             severity=1.5, phrases=["a"])


def test_rule_validation_requires_phrases():
    with pytest.raises(ValueError):
        Rule(id="x", category=Category.SPAM, rule_type=RuleType.KEYWORD,
             severity=0.5, phrases=[])


def test_rule_serializable_as_data():
    # Rules are plain dataclasses -> introspectable as data (the DSL property).
    rule = _rule("pii.email")
    assert rule.category == Category.PII
    assert rule.detector == "email"
    assert rule.rule_type == RuleType.PII
