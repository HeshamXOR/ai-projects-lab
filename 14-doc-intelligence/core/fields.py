"""Layout-aware field extractor.

Extracts the scalar header fields of an invoice / receipt from raw or OCR'd
text: vendor, invoice number, date, currency and the monetary totals
(subtotal / tax / total, with an "amount due" precedence rule).

The extractor is *layout aware* in the sense that it uses positional hints --
which line a value appears on, whether it sits in the top-of-document block,
and how a value is aligned relative to its label -- rather than treating the
document as a bag of words. Everything is hand-written heuristics; there is no
model dependency.

Each extractor returns an :class:`Extraction` carrying the value plus the list
of *signals* that fired. The confidence layer consumes those signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from decimal import Decimal
from typing import List, Optional

from .normalize import (
    CURRENCY_CODES,
    CURRENCY_SYMBOLS,
    detect_currency,
    normalize_amount,
    normalize_date,
)

__all__ = [
    "Extraction",
    "extract_vendor",
    "extract_invoice_number",
    "extract_date",
    "extract_currency",
    "extract_totals",
    "Totals",
]


@dataclass
class Extraction:
    """A single extracted field with the signals that supported it."""

    value: Optional[object]
    raw: Optional[str] = None
    signals: List[str] = dc_field(default_factory=list)

    def add(self, signal: str) -> "Extraction":
        self.signals.append(signal)
        return self


# --------------------------------------------------------------------------- #
# Vendor
# --------------------------------------------------------------------------- #

_COMPANY_SUFFIXES = (
    "inc", "inc.", "llc", "l.l.c.", "ltd", "ltd.", "limited", "corp", "corp.",
    "co", "co.", "company", "gmbh", "ag", "plc", "llp", "pvt", "pty",
    "s.a.", "sarl", "bv", "srl", "group", "studio", "studios", "solutions",
    "services", "consulting", "associates", "partners", "enterprises",
    "systems", "technologies", "labs",
)

_VENDOR_STOP_LINES = (
    "invoice", "receipt", "bill to", "ship to", "sold to", "statement",
    "tax invoice", "estimate", "quote", "purchase order",
)


def _looks_like_address(line: str) -> bool:
    low = line.lower()
    if re.search(r"\b\d{3,}\b", line) and any(
        w in low for w in ("street", "st.", "ave", "avenue", "road", "rd", "suite", "blvd")
    ):
        return True
    if re.match(r"^\s*\d+\s+\w+", line):  # starts with a street number
        return True
    return False


def extract_vendor(text: str) -> Extraction:
    """Find the vendor name from the top-of-document block.

    Heuristics (each contributes a signal):
      * Look only at the first handful of non-empty lines (top block).
      * Prefer a line containing a company suffix (Inc, LLC, Ltd, ...).
      * Otherwise prefer the first ALL-CAPS or Title-Case line that is not a
        document keyword ("INVOICE") and does not look like an address.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    top = [ln for ln in lines[:8] if ln]
    ext = Extraction(value=None)

    # Pass 1: company suffix in the top block.
    for ln in top:
        words = re.findall(r"[A-Za-z&.]+", ln)
        if not words:
            continue
        low_words = [w.lower().strip(".") for w in words]
        if any(w in {s.strip(".") for s in _COMPANY_SUFFIXES} for w in low_words):
            if not any(k in ln.lower() for k in _VENDOR_STOP_LINES):
                ext.value = ln.strip(" :|-")
                ext.raw = ln
                ext.add("company_suffix_present").add("in_top_block")
                return ext

    # Pass 2: first prominent non-keyword line.
    for ln in top:
        low = ln.lower()
        if any(k in low for k in _VENDOR_STOP_LINES):
            continue
        if _looks_like_address(ln):
            continue
        letters = [c for c in ln if c.isalpha()]
        if len(letters) < 2:
            continue
        is_caps = ln.isupper()
        is_title = ln == ln.title() or ln.split()[0][:1].isupper()
        if is_caps or is_title:
            ext.value = ln.strip(" :|-")
            ext.raw = ln
            ext.add("in_top_block")
            if is_caps:
                ext.add("all_caps_heading")
            elif is_title:
                ext.add("title_case_heading")
            return ext

    # Fallback: very first non-empty line.
    if top:
        ext.value = top[0].strip(" :|-")
        ext.raw = top[0]
        ext.add("first_line_fallback")
    return ext


# --------------------------------------------------------------------------- #
# Invoice number
# --------------------------------------------------------------------------- #

# Require either an explicit number-ish qualifier (#, no, number, id) or a
# ':' after the label, so the bare word "INVOICE" alone does not swallow the
# following token (e.g. capturing "OICE" out of "INVOICE").
_INV_LABEL = re.compile(
    r"\b(?:invoice|inv|bill|receipt|ref(?:erence)?|rechnung|rechnungs[- ]?nr)\b\s*"
    r"(?:(?:#|no\.?|nr\.?|number|num\.?|id)\s*[:#]?\s*|[:#]\s*)"
    r"([A-Za-z0-9][A-Za-z0-9\-_/]{1,30})",
    re.IGNORECASE,
)
_HASH_NUMBER = re.compile(r"#\s*([A-Za-z0-9][A-Za-z0-9\-_/]{1,30})")


def extract_invoice_number(text: str) -> Extraction:
    """Extract the invoice / receipt number."""
    ext = Extraction(value=None)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if any(k in low for k in ("invoice", "inv", "bill", "receipt", "ref", "rechnung")):
            m = _INV_LABEL.search(line)
            if m:
                candidate = m.group(1).strip(" .:-")
                # Reject pure-month words accidentally captured.
                if candidate and not candidate.lower() in ("to", "from", "date", "for"):
                    ext.value = candidate
                    ext.raw = line
                    ext.add("label_keyword_present")
                    if re.search(r"\d", candidate):
                        ext.add("contains_digits")
                    return ext
    # Fallback: a "#1234" style token anywhere.
    m = _HASH_NUMBER.search(text)
    if m:
        ext.value = m.group(1).strip()
        ext.raw = m.group(0)
        ext.add("hash_number_pattern")
    return ext


# --------------------------------------------------------------------------- #
# Date
# --------------------------------------------------------------------------- #

_DATE_LABEL = re.compile(r"(invoice date|date of issue|issued|date due|due date|datum|date)\s*[:#]?\s*(.+)", re.IGNORECASE)
_DATE_CANDIDATE = re.compile(
    r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}[\s\-/]+[A-Za-z]{3,9}[\s\-/,]+\d{2,4}"
    r"|[A-Za-z]{3,9}[\s\-/]+\d{1,2}(?:st|nd|rd|th)?[\s\-/,]+\d{2,4})"
)


def extract_date(text: str, locale_hint: Optional[str] = None) -> Extraction:
    """Extract and normalize the primary invoice date to ISO-8601.

    Labelled dates ("Invoice Date: ...") are preferred over loose dates.
    """
    ext = Extraction(value=None)

    # Pass 1: labelled date lines.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = _DATE_LABEL.match(line)
        if m:
            tail = m.group(2)
            cand = _DATE_CANDIDATE.search(tail)
            if cand:
                iso = normalize_date(cand.group(1), locale_hint)
                if iso:
                    ext.value = iso
                    ext.raw = cand.group(1)
                    ext.add("label_keyword_present").add("format_matched")
                    return ext

    # Pass 2: first parseable date anywhere.
    for cand in _DATE_CANDIDATE.finditer(text):
        iso = normalize_date(cand.group(1), locale_hint)
        if iso:
            ext.value = iso
            ext.raw = cand.group(1)
            ext.add("format_matched")
            return ext
    return ext


# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #


def extract_currency(text: str) -> Extraction:
    """Detect the document currency from symbols / explicit codes."""
    ext = Extraction(value=None)
    code = detect_currency(text)
    if code:
        ext.value = code
        ext.add("currency_detected")
        # Stronger signal if an explicit ISO code is present.
        if re.search(r"\b" + "|".join(CURRENCY_CODES) + r"\b", text):
            ext.add("explicit_iso_code")
        for sym in CURRENCY_SYMBOLS:
            if sym in text:
                ext.add("symbol_present")
                break
    return ext


# --------------------------------------------------------------------------- #
# Totals (subtotal / tax / total) with amount-due precedence
# --------------------------------------------------------------------------- #


@dataclass
class Totals:
    subtotal: Optional[Extraction] = None
    tax: Optional[Extraction] = None
    total: Optional[Extraction] = None


# Label -> which bucket it feeds. Order matters for "total" precedence.
_TOTAL_PATTERNS = [
    ("subtotal", re.compile(r"\b(sub[\s-]?total|zwischensumme)\b", re.IGNORECASE)),
    ("tax", re.compile(r"\b(tax|vat|gst|sales tax|mwst|ust|umsatzsteuer)\b", re.IGNORECASE)),
    ("amount_due", re.compile(
        r"\b(amount due|balance due|total due|amount payable|total payable"
        r"|gesamtbetrag|gesamtsumme|zu zahlen)\b", re.IGNORECASE)),
    ("total", re.compile(r"\b(grand total|total|invoice total|summe)\b", re.IGNORECASE)),
]

# Find the rightmost money value on a line (totals are right-aligned).
_LINE_MONEY = re.compile(
    r"(?P<sym>[$€£¥₹₩₽])?\s*"
    r"(?P<num>\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"\s*(?P<code>USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY|KRW)?"
)


def _rightmost_amount(line: str, default_currency: Optional[str]) -> Optional[Extraction]:
    matches = list(_LINE_MONEY.finditer(line))
    if not matches:
        return None
    m = matches[-1]
    parsed = normalize_amount(m.group(0), default_currency)
    if parsed is None:
        return None
    ext = Extraction(value=parsed, raw=m.group(0))
    ext.add("format_matched")
    return ext


def extract_totals(text: str, default_currency: Optional[str] = None) -> Totals:
    """Extract subtotal, tax and total.

    Precedence rule: an explicit "Amount Due" / "Balance Due" beats a plain
    "Total" for the *total* bucket, because vendors frequently print both a
    line total and a final amount-due that includes tax / prior balance.
    """
    totals = Totals()
    found_amount_due = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        for bucket, pat in _TOTAL_PATTERNS:
            if not pat.search(low):
                continue
            amt = _rightmost_amount(line, default_currency)
            if amt is None:
                continue
            amt.add("label_keyword_present")

            if bucket == "subtotal":
                totals.subtotal = amt
            elif bucket == "tax":
                totals.tax = amt
            elif bucket == "amount_due":
                totals.total = amt
                amt.add("amount_due_precedence")
                found_amount_due = True
            elif bucket == "total":
                # Do not let a plain "total" overwrite an amount-due value.
                if not found_amount_due:
                    # Avoid matching "subtotal" via the "total" substring: the
                    # subtotal pattern already consumed it on this line only if
                    # the line is literally a subtotal line.
                    if "subtotal" in low or "sub total" in low or "sub-total" in low:
                        continue
                    totals.total = amt
            break  # one bucket per line
    return totals
