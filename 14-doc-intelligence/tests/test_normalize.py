"""Tests for the from-scratch normalization layer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.normalize import (
    NormalizedAmount,
    is_leap_year,
    normalize_amount,
    normalize_date,
)


class TestDateNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-01-31", "2024-01-31"),
            ("2024/02/29", "2024-02-29"),       # leap year valid
            ("31/01/2024", "2024-01-31"),       # day-first numeric
            ("01/31/2024", "2024-01-31"),       # unambiguous (31 = day)
            ("12 January 2024", "2024-01-12"),
            ("12 Jan 2024", "2024-01-12"),
            ("Jan 12, 2024", "2024-01-12"),
            ("January 12, 2024", "2024-01-12"),
            ("12-Feb-24", "2024-02-12"),
            ("14 Februar 2024", "2024-02-14"),  # German month
            ("3.15.2024", "2024-03-15"),        # dotted, unambiguous-ish
        ],
    )
    def test_formats(self, raw, expected):
        assert normalize_date(raw) == expected

    def test_locale_disambiguation(self):
        # 01/02/2024 is ambiguous; locale decides.
        assert normalize_date("01/02/2024", "en_US") == "2024-01-02"  # m/d
        assert normalize_date("01/02/2024", "en_GB") == "2024-02-01"  # d/m
        assert normalize_date("01/02/2024") == "2024-02-01"           # default d/m

    @pytest.mark.parametrize(
        "raw",
        ["2024-02-30", "31/13/2024", "not a date", "", "99/99/9999"],
    )
    def test_invalid_returns_none(self, raw):
        assert normalize_date(raw) is None

    def test_leap_year(self):
        assert is_leap_year(2024)
        assert is_leap_year(2000)
        assert not is_leap_year(1900)
        assert not is_leap_year(2023)
        # Feb 29 only valid on leap years.
        assert normalize_date("2023-02-29") is None
        assert normalize_date("2024-02-29") == "2024-02-29"

    def test_two_digit_year_window(self):
        assert normalize_date("01/01/68") == "2068-01-01"
        assert normalize_date("01/01/69") == "1969-01-01"


class TestAmountNormalization:
    def test_anglo_convention(self):
        amt = normalize_amount("$1,234.56")
        assert amt is not None
        assert amt.value == Decimal("1234.56")
        assert amt.currency == "USD"

    def test_european_convention(self):
        amt = normalize_amount("1.234,56 EUR")
        assert amt is not None
        assert amt.value == Decimal("1234.56")
        assert amt.currency == "EUR"

    def test_european_symbol(self):
        amt = normalize_amount("€1.234,56")
        assert amt.value == Decimal("1234.56")
        assert amt.currency == "EUR"

    @pytest.mark.parametrize(
        "raw,value",
        [
            ("12,5", Decimal("12.5")),       # european decimal, no thousands
            ("12.50", Decimal("12.50")),     # anglo decimal
            ("1,000", Decimal("1000")),      # anglo thousands
            ("1.000", Decimal("1000")),      # european thousands
            ("1,234,567.89", Decimal("1234567.89")),
            ("1.234.567,89", Decimal("1234567.89")),
            ("0.00", Decimal("0.00")),
            ("£95.00", Decimal("95.00")),
        ],
    )
    def test_separator_conventions(self, raw, value):
        amt = normalize_amount(raw)
        assert amt is not None
        assert amt.value == value

    def test_currency_codes(self):
        assert normalize_amount("£760.00").currency == "GBP"
        assert normalize_amount("¥5000").currency == "JPY"
        assert normalize_amount("100.00").currency is None

    def test_default_currency(self):
        amt = normalize_amount("100.00", default_currency="CAD")
        assert amt.currency == "CAD"

    def test_negative(self):
        amt = normalize_amount("-$50.00")
        assert amt.value == Decimal("-50.00")

    @pytest.mark.parametrize("raw", ["", None, "abc", "$"])
    def test_unparseable(self, raw):
        assert normalize_amount(raw) is None

    def test_str_repr(self):
        assert str(NormalizedAmount(Decimal("10.00"), "USD")) == "10.00 USD"
        assert str(NormalizedAmount(Decimal("10.00"))) == "10.00"
