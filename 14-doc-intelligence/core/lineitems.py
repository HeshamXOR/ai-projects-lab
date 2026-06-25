"""From-scratch line-item parser for invoice tables.

This is the centerpiece of the project. There is **no** machine-learning model
here -- it is a hand-built tokenizer feeding a rule-based state machine that:

1. Tokenizes each candidate row into typed tokens (WORD, MONEY, NUMBER,
   PERCENT, SEPARATOR).
2. Classifies the *numeric column shape* of the row. Invoice line items
   overwhelmingly fall into a small number of column layouts:

   * ``desc qty unit_price amount``      (4 numeric-ish trailing fields)
   * ``desc qty amount``                 (qty + a single money column)
   * ``desc amount``                     (description + a single money column)
   * ``desc qty unit_price`` with amount implied

3. Aligns the trailing numeric tokens to the (qty, unit_price, line_total)
   slots using positional heuristics plus an **arithmetic consistency check**
   (``qty * unit_price approx line_total``). The arithmetic check is what lets
   the state machine *recover* the correct alignment when a row is ambiguous --
   e.g. distinguishing a unit price from a discount column.

The public entry point is :func:`parse_line_items`, which scans a full block of
text, decides which lines look like item rows (vs. headers / totals / address
blocks) and returns a list of :class:`LineItem`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Tuple

from .normalize import normalize_amount

__all__ = [
    "TokenType",
    "Token",
    "LineItem",
    "tokenize_row",
    "classify_row",
    "parse_line_items",
]


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


class TokenType(Enum):
    MONEY = "MONEY"          # has a currency symbol/code
    NUMBER = "NUMBER"        # bare numeric (could be qty or price)
    PERCENT = "PERCENT"      # 7.5%
    WORD = "WORD"            # part of the description
    SEPARATOR = "SEPARATOR"  # column gap / pipe / tab


@dataclass
class Token:
    type: TokenType
    text: str
    value: Optional[Decimal] = None  # populated for MONEY / NUMBER / PERCENT
    has_currency: bool = False


# A number, optionally with currency symbol, thousands separators and decimals.
_MONEY_RE = re.compile(
    r"""
    (?P<neg>-|\()?              # optional negative / open paren
    \s*
    (?P<sym>[$€£¥₹₩₽])?          # optional currency symbol
    \s*
    (?P<num>\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)
    \s*
    (?P<code>USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY|KRW)?
    (?P<closeparen>\))?
    """,
    re.VERBOSE,
)

_PERCENT_RE = re.compile(r"^\d+(?:[.,]\d+)?%$")
_CURRENCY_HINT = re.compile(r"[$€£¥₹₩₽]|\b(?:USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY|KRW)\b")


def _split_columns(line: str) -> List[str]:
    """Split a row into column-ish chunks.

    We prefer explicit column delimiters (tabs, pipes, runs of >=2 spaces)
    because invoices are usually laid out in columns. We then keep the chunks
    so that the classifier can reason about column *positions*.
    """
    # Normalise pipes/tabs to a double-space so one splitter handles all.
    norm = line.replace("\t", "  ").replace("|", "  ")
    parts = re.split(r"\s{2,}", norm.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize_row(line: str) -> List[Token]:
    """Tokenize a single physical line into typed tokens."""
    tokens: List[Token] = []
    columns = _split_columns(line)
    for col in columns:
        # Within a column we still split on single spaces so a description like
        # "Web design services" stays as WORD tokens but a trailing "2 50.00"
        # becomes two numeric tokens.
        for raw in col.split(" "):
            raw = raw.strip()
            if not raw:
                continue
            tok = _classify_token(raw)
            tokens.append(tok)
        tokens.append(Token(TokenType.SEPARATOR, "  "))
    if tokens and tokens[-1].type is TokenType.SEPARATOR:
        tokens.pop()
    return tokens


def _classify_token(raw: str) -> Token:
    if _PERCENT_RE.match(raw):
        num = normalize_amount(raw.rstrip("%"))
        return Token(TokenType.PERCENT, raw, num.value if num else None)

    has_cur = bool(_CURRENCY_HINT.search(raw))
    # Attempt a money/number parse on the whole token.
    m = _MONEY_RE.fullmatch(raw.strip())
    if m and m.group("num"):
        parsed = normalize_amount(raw)
        if parsed is not None:
            ttype = TokenType.MONEY if has_cur else TokenType.NUMBER
            return Token(ttype, raw, parsed.value, has_currency=has_cur)
    return Token(TokenType.WORD, raw)


# --------------------------------------------------------------------------- #
# Row classification + the alignment state machine
# --------------------------------------------------------------------------- #


@dataclass
class LineItem:
    description: str
    quantity: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    line_total: Optional[Decimal] = None
    currency: Optional[str] = None
    arithmetic_ok: bool = False
    raw: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def dec(x: Optional[Decimal]):
            return str(x) if x is not None else None

        return {
            "description": self.description,
            "quantity": dec(self.quantity),
            "unit_price": dec(self.unit_price),
            "line_total": dec(self.line_total),
            "currency": self.currency,
            "arithmetic_ok": self.arithmetic_ok,
        }


# Keywords that strongly indicate a line is NOT an item row.
_NON_ITEM_KEYWORDS = (
    "subtotal", "sub total", "sub-total", "total", "amount due", "balance",
    "tax", "vat", "gst", "shipping", "discount applied", "invoice", "date",
    "bill to", "ship to", "payment", "thank you", "terms", "due date",
    "account", "routing", "remit",
    # German
    "zwischensumme", "gesamtbetrag", "gesamtsumme", "mwst", "ust",
    "umsatzsteuer", "rechnung", "datum", "summe",
)
# Header rows for the items table.
_HEADER_KEYWORDS = (
    "description", "qty", "quantity", "unit", "price", "amount", "item",
    "beschreibung", "menge", "einzelpreis", "betrag",
)


def _approx_equal(a: Decimal, b: Decimal, rel: Decimal = Decimal("0.02")) -> bool:
    """qty*price vs total within 2% (or 1 cent absolute) tolerance."""
    diff = abs(a - b)
    if diff <= Decimal("0.01"):
        return True
    scale = max(abs(a), abs(b), Decimal("1"))
    return diff / scale <= rel


def classify_row(tokens: List[Token]) -> Tuple[str, List[Token], List[Token]]:
    """Split a row's tokens into (description tokens, numeric tokens).

    Returns a tuple ``(shape, desc_tokens, numeric_tokens)`` where *shape* is a
    short string describing the trailing numeric column count.
    """
    numeric = [t for t in tokens if t.type in (TokenType.MONEY, TokenType.NUMBER)]
    # Description = the leading WORD/SEPARATOR run before the first numeric col
    # that is part of the trailing numeric block. We take the trailing numerics
    # (those that appear after the last WORD token).
    last_word_idx = -1
    for i, t in enumerate(tokens):
        if t.type is TokenType.WORD:
            last_word_idx = i

    trailing_numeric = [
        t for t in tokens[last_word_idx + 1:]
        if t.type in (TokenType.MONEY, TokenType.NUMBER, TokenType.PERCENT)
    ]
    desc_tokens = [t for t in tokens[: last_word_idx + 1] if t.type is TokenType.WORD]

    n = len([t for t in trailing_numeric if t.type in (TokenType.MONEY, TokenType.NUMBER)])
    if n >= 4:
        shape = "quad+"
    elif n == 3:
        shape = "triple"
    elif n == 2:
        shape = "double"
    elif n == 1:
        shape = "single"
    else:
        shape = "none"
    return shape, desc_tokens, numeric if not trailing_numeric else trailing_numeric


# Tokens that indicate a postal address line rather than a priced item.
# Matched on word boundaries so "rd" does not fire inside "Sourdough".
_ADDRESS_KEYWORDS = (
    "street", "st", "avenue", "ave", "road", "rd", "suite", "ste",
    "blvd", "boulevard", "lane", "ln", "drive", "floor", "fl",
    "apt", "box", "strasse", "straße", "str",
)
_ADDRESS_KW_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _ADDRESS_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
# US state abbreviations + a 5-digit ZIP strongly imply an address line.
_STATE_ZIP_RE = re.compile(r"\b[A-Z]{2}\b[ ,]+\d{5}\b")
# A German-style "Strasse"/"straße" suffixed directly onto a word, plus a
# trailing house number, e.g. "Hauptstraße 17".
_DE_STREET_RE = re.compile(r"\w+(?:stra(?:ss|ß)e|str\.)\s+\d{1,4}\b", re.IGNORECASE)


def _looks_like_address_row(line: str) -> bool:
    if _ADDRESS_KW_RE.search(line):
        return True
    if _STATE_ZIP_RE.search(line):
        return True
    if _DE_STREET_RE.search(line):
        return True
    # Starts with a street number followed by words (e.g. "123 Market ...").
    if re.match(r"^\s*\d{1,6}\s+[A-Za-z]", line) and not re.search(r"[$€£¥₹]", line):
        # but only if there is no decimal money column (real items have prices)
        if not re.search(r"\d[.,]\d{2}\b", line):
            return True
    return False


def _is_item_candidate(line: str, tokens: List[Token]) -> bool:
    low = line.lower()
    for kw in _NON_ITEM_KEYWORDS:
        if kw in low:
            return False
    if _looks_like_address_row(line):
        return False
    # Pure header row.
    word_text = " ".join(t.text.lower() for t in tokens if t.type is TokenType.WORD)
    if word_text and all(
        any(h in w for h in _HEADER_KEYWORDS) for w in word_text.split()
    ):
        return False
    # Must have at least one numeric token AND at least one descriptive word.
    has_num = any(t.type in (TokenType.MONEY, TokenType.NUMBER) for t in tokens)
    has_word = any(t.type is TokenType.WORD for t in tokens)
    return has_num and has_word


def _align(
    shape: str, numeric: List[Token]
) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal], bool, List[str]]:
    """The alignment state machine.

    Given the row shape and trailing numeric tokens, decide which token maps to
    quantity, unit_price and line_total, using positional defaults and then an
    arithmetic check to validate / repair the guess.

    Returns ``(qty, unit_price, line_total, arithmetic_ok, notes)``.
    """
    notes: List[str] = []
    nums = [t for t in numeric if t.type in (TokenType.MONEY, TokenType.NUMBER)]
    vals = [t.value for t in nums]

    qty = unit = total = None
    ok = False

    if shape == "quad+":
        # Take the last 3 money-ish columns as price/total and the first plain
        # number as qty. Common layout: desc | qty | unit | (tax) | amount.
        qty = vals[0]
        unit = vals[1]
        total = vals[-1]
        if qty is not None and unit is not None and total is not None:
            ok = _approx_equal(qty * unit, total)
            if not ok:
                # try unit = second-to-last
                alt_unit = vals[-2]
                if _approx_equal(qty * alt_unit, total):
                    unit = alt_unit
                    ok = True
                    notes.append("re-aligned unit price from extra column")
    elif shape == "triple":
        # desc qty unit total
        qty, unit, total = vals[0], vals[1], vals[2]
        ok = _approx_equal(qty * unit, total)
        if not ok:
            # Maybe order is qty total unit, or unit qty total -> brute force.
            for perm in (
                (vals[0], vals[1], vals[2]),
                (vals[0], vals[2], vals[1]),
                (vals[1], vals[0], vals[2]),
            ):
                q, u, t = perm
                if q is not None and u is not None and t is not None and _approx_equal(q * u, t):
                    qty, unit, total, ok = q, u, t, True
                    notes.append("re-ordered numeric columns via arithmetic check")
                    break
    elif shape == "double":
        # Two candidates: most often (qty, amount) where unit==amount/qty,
        # or (unit_price, amount) with implicit qty 1.
        a, b = vals[0], vals[1]
        total = b
        # Heuristic: if a is a small whole number it's probably a quantity.
        if a is not None and a == a.to_integral_value() and a <= 1000 and b is not None and a != 0:
            qty = a
            unit = (b / a) if a != 0 else None
            if unit is not None:
                unit = unit.quantize(Decimal("0.01"))
                ok = _approx_equal(qty * unit, total)
            notes.append("interpreted as (qty, amount); derived unit price")
        else:
            qty = Decimal("1")
            unit = a
            total = b
            ok = _approx_equal(unit, total) if unit is not None else False
    elif shape == "single":
        total = vals[0]
        qty = Decimal("1")
        unit = vals[0]
        ok = True
        notes.append("single amount column; assumed qty 1")

    return qty, unit, total, ok, notes


def parse_line_items(text: str) -> List[LineItem]:
    """Scan *text*, find item rows, and return structured :class:`LineItem`s."""
    items: List[LineItem] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        tokens = tokenize_row(line)
        if not _is_item_candidate(line, tokens):
            continue

        shape, desc_tokens, numeric = classify_row(tokens)
        if shape == "none":
            continue

        qty, unit, total, ok, notes = _align(shape, numeric)
        if total is None:
            continue

        description = " ".join(t.text for t in desc_tokens).strip(" -:|")
        if not description:
            # Probably a totals line that slipped through; skip.
            continue

        currency = None
        for t in numeric:
            if t.has_currency:
                parsed = normalize_amount(t.text)
                if parsed and parsed.currency:
                    currency = parsed.currency
                    break

        items.append(
            LineItem(
                description=description,
                quantity=qty,
                unit_price=unit,
                line_total=total,
                currency=currency,
                arithmetic_ok=ok,
                raw=line,
                notes=notes,
            )
        )
    return items
