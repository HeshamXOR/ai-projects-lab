"""Tests for the from-scratch line-item state machine."""

from __future__ import annotations

from decimal import Decimal

from core.lineitems import (
    TokenType,
    classify_row,
    parse_line_items,
    tokenize_row,
)


class TestTokenizer:
    def test_typed_tokens(self):
        toks = tokenize_row("Web design services    10    150.00    1500.00")
        types = [t.type for t in toks if t.type is not TokenType.SEPARATOR]
        assert TokenType.WORD in types
        assert TokenType.NUMBER in types
        nums = [t.value for t in toks if t.type is TokenType.NUMBER]
        assert Decimal("10") in nums
        assert Decimal("1500.00") in nums

    def test_money_token_currency(self):
        toks = tokenize_row("Consulting    5    $200.00    $1,000.00")
        money = [t for t in toks if t.type is TokenType.MONEY]
        assert money, "expected MONEY tokens for $-prefixed values"
        assert any(t.value == Decimal("1000.00") for t in money)


class TestRowClassification:
    def test_quad_shape(self):
        toks = tokenize_row("Item desc    2    10.00    1.50    21.50")
        shape, desc, numeric = classify_row(toks)
        assert shape == "quad+"

    def test_triple_shape(self):
        toks = tokenize_row("Item    10    150.00    1500.00")
        shape, _, _ = classify_row(toks)
        assert shape == "triple"

    def test_single_shape(self):
        toks = tokenize_row("Domain registration    14.99")
        shape, _, _ = classify_row(toks)
        assert shape == "single"


class TestLineItemParsing:
    def test_basic_table(self):
        text = (
            "Web design services    10    150.00    1500.00\n"
            "Annual hosting         1     240.00    240.00\n"
            "Consulting             5     200.00    1000.00\n"
        )
        items = parse_line_items(text)
        assert len(items) == 3
        first = items[0]
        assert first.quantity == Decimal("10")
        assert first.unit_price == Decimal("150.00")
        assert first.line_total == Decimal("1500.00")
        assert first.arithmetic_ok is True

    def test_arithmetic_validation(self):
        # 10 * 150 == 1500 -> ok
        items = parse_line_items("Service A    10    150.00    1500.00")
        assert items[0].arithmetic_ok is True

    def test_arithmetic_mismatch_flagged(self):
        # 3 * 10 != 999 ; should still parse but flag arithmetic_ok False
        items = parse_line_items("Bad row    3    10.00    999.00")
        assert len(items) == 1
        assert items[0].arithmetic_ok is False

    def test_skips_totals_and_headers(self):
        text = (
            "Description    Qty    Unit Price    Amount\n"
            "Widget         2      5.00          10.00\n"
            "Subtotal                            10.00\n"
            "Tax                                 0.80\n"
            "Total                               10.80\n"
        )
        items = parse_line_items(text)
        descriptions = [it.description.lower() for it in items]
        assert any("widget" in d for d in descriptions)
        assert not any("subtotal" in d for d in descriptions)
        assert not any("total" in d for d in descriptions)
        assert not any("tax" in d for d in descriptions)
        assert len(items) == 1

    def test_skips_address_lines(self):
        text = (
            "123 Market Street, Suite 400\n"
            "San Francisco, CA 94103\n"
            "Widget    2    5.00    10.00\n"
        )
        items = parse_line_items(text)
        assert len(items) == 1
        assert "widget" in items[0].description.lower()

    def test_single_amount_row(self):
        items = parse_line_items("Domain registration    14.99")
        assert len(items) == 1
        assert items[0].line_total == Decimal("14.99")
        assert items[0].quantity == Decimal("1")

    def test_european_amounts(self):
        items = parse_line_items("Monitor    1    1.234,56    1.234,56")
        assert len(items) == 1
        assert items[0].line_total == Decimal("1234.56")
        assert items[0].arithmetic_ok is True

    def test_qty_amount_derives_unit_price(self):
        # double shape: (qty, amount) -> derive unit price
        items = parse_line_items("Apples    4    11.96")
        assert len(items) == 1
        it = items[0]
        assert it.quantity == Decimal("4")
        assert it.unit_price == Decimal("2.99")
        assert it.line_total == Decimal("11.96")
