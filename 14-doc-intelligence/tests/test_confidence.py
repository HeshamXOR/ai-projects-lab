"""Tests for the confidence model."""

from __future__ import annotations

from core.confidence import (
    aggregate_confidence,
    score_field,
    score_line_item,
    score_signals,
)


class TestSignalScoring:
    def test_empty_is_zero(self):
        assert score_field([]) == 0.0

    def test_bounded_zero_one(self):
        score = score_signals(["label_keyword_present", "format_matched"])
        assert 0.0 < score < 1.0

    def test_monotonic_more_signals_higher(self):
        # Adding corroborating signals must never lower the score.
        s1 = score_signals(["label_keyword_present"])
        s2 = score_signals(["label_keyword_present", "format_matched"])
        s3 = score_signals(
            ["label_keyword_present", "format_matched", "contains_digits"]
        )
        assert s1 < s2 < s3

    def test_duplicate_signals_no_inflation(self):
        s_once = score_signals(["format_matched"])
        s_twice = score_signals(["format_matched", "format_matched"])
        assert s_once == s_twice

    def test_saturation_never_exceeds_one(self):
        many = ["label_keyword_present", "format_matched", "company_suffix_present",
                "in_top_block", "explicit_iso_code", "amount_due_precedence"]
        assert score_signals(many) < 1.0

    def test_unknown_signal_uses_default(self):
        assert score_signals(["totally_unknown_signal"]) > 0.0


class TestLineItemConfidence:
    def test_arithmetic_increases_confidence(self):
        with_ok = score_line_item(
            arithmetic_ok=True, has_quantity=True, has_unit_price=True,
            has_currency=True,
        )
        without_ok = score_line_item(
            arithmetic_ok=False, has_quantity=True, has_unit_price=True,
            has_currency=True,
        )
        assert with_ok > without_ok

    def test_more_fields_more_confidence(self):
        sparse = score_line_item(
            arithmetic_ok=False, has_quantity=False, has_unit_price=False,
            has_currency=False,
        )
        rich = score_line_item(
            arithmetic_ok=True, has_quantity=True, has_unit_price=True,
            has_currency=True,
        )
        assert rich > sparse


class TestAggregate:
    def test_mean_of_fields(self):
        agg = aggregate_confidence({"a": 0.5, "b": 0.5})
        assert agg == 0.5

    def test_empty_is_zero(self):
        assert aggregate_confidence({}) == 0.0
