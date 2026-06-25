"""Normalization layer for dates and monetary amounts.

Everything here is implemented from scratch using only the Python standard
library. In particular:

* Dates are parsed without ``dateutil`` -- we run a battery of hand-written
  format matchers (numeric ``dd/mm/yyyy`` / ``mm/dd/yyyy`` ambiguity handling,
  ISO, and "12 January 2024" / "Jan 12, 2024" style textual dates) and emit
  ISO-8601 (``YYYY-MM-DD``).
* Amounts are parsed from raw strings that may carry currency symbols,
  thousands separators and either the Anglo ``1,234.56`` or the European
  ``1.234,56`` convention. We disambiguate the two conventions structurally
  rather than guessing, and return a :class:`decimal.Decimal` plus an ISO-4217
  currency code when one can be inferred.

The functions return ``None`` (never raise) when the input cannot be parsed so
that callers / the confidence layer can treat a failed parse as a weak signal
rather than an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

__all__ = [
    "MONTHS",
    "CURRENCY_SYMBOLS",
    "CURRENCY_CODES",
    "NormalizedAmount",
    "normalize_date",
    "normalize_amount",
    "detect_currency",
    "is_leap_year",
]

# --------------------------------------------------------------------------- #
# Calendar helpers
# --------------------------------------------------------------------------- #

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
    # German month names (common on EU invoices).
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "mai": 5,
    "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}

_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap_year(year: int) -> bool:
    """Return True if *year* is a Gregorian leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    if month == 2 and is_leap_year(year):
        return 29
    return _DAYS_IN_MONTH[month - 1]


def _valid_ymd(year: int, month: int, day: int) -> bool:
    if not (1 <= month <= 12):
        return False
    if not (1 <= day <= _days_in_month(year, month)):
        return False
    if not (1 <= year <= 9999):
        return False
    return True


def _expand_year(year: int) -> int:
    """Expand a 2-digit year to 4 digits with a sliding window.

    Years 0-68 -> 2000-2068, 69-99 -> 1969-1999. This mirrors the POSIX
    strptime ``%y`` convention which is a reasonable default for invoices.
    """
    if year >= 100:
        return year
    if year <= 68:
        return 2000 + year
    return 1900 + year


# --------------------------------------------------------------------------- #
# Date normalization
# --------------------------------------------------------------------------- #

# ISO-ish: 2024-01-31 or 2024/01/31
_RE_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
# Numeric d/m/y or m/d/y with 1-4 digit year
_RE_NUMERIC = re.compile(r"^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})$")
# 12 January 2024  /  12 Jan 2024  /  12-Jan-2024
_RE_DMY_TEXT = re.compile(
    r"^(\d{1,2})[\s\-/]+([A-Za-z]{3,9})[\s\-/,]+(\d{2,4})$"
)
# January 12, 2024 / Jan 12 2024
_RE_MDY_TEXT = re.compile(
    r"^([A-Za-z]{3,9})[\s\-/]+(\d{1,2})(?:st|nd|rd|th)?[\s\-/,]+(\d{2,4})$"
)


def _try_iso(text: str) -> Optional[str]:
    m = _RE_ISO.match(text)
    if not m:
        return None
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if _valid_ymd(year, month, day):
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _try_numeric(text: str, prefer_day_first: bool) -> Optional[str]:
    """Parse purely numeric dates, handling d/m/y vs m/d/y ambiguity.

    Strategy:
      * If the first component is clearly a 4-digit year, treat as y/m/d.
      * If one of the first two components is > 12 it *must* be the day, so we
        can resolve unambiguously regardless of locale.
      * Otherwise fall back to the locale hint (*prefer_day_first*).
    """
    m = _RE_NUMERIC.match(text)
    if not m:
        return None
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Leading 4-digit year -> y/m/d (already handled by ISO, but accept here too)
    if m.group(1).__len__() == 4:
        year, month, day = a, b, c
        if _valid_ymd(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None

    year = _expand_year(c)

    # Structural disambiguation of the first two fields.
    if a > 12 and b <= 12:
        day, month = a, b
    elif b > 12 and a <= 12:
        month, day = a, b
    else:
        # Genuinely ambiguous -> use locale hint.
        if prefer_day_first:
            day, month = a, b
        else:
            month, day = a, b

    if _valid_ymd(year, month, day):
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _try_textual(text: str) -> Optional[str]:
    m = _RE_DMY_TEXT.match(text)
    if m:
        day = int(m.group(1))
        month = MONTHS.get(m.group(2).lower())
        year = _expand_year(int(m.group(3)))
        if month and _valid_ymd(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"

    m = _RE_MDY_TEXT.match(text)
    if m:
        month = MONTHS.get(m.group(1).lower())
        day = int(m.group(2))
        year = _expand_year(int(m.group(3)))
        if month and _valid_ymd(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def normalize_date(raw: str, locale_hint: Optional[str] = None) -> Optional[str]:
    """Normalize a date string to ISO-8601 ``YYYY-MM-DD``.

    Parameters
    ----------
    raw:
        The raw date text, e.g. ``"31/01/2024"``, ``"Jan 12, 2024"``.
    locale_hint:
        Optional locale string. Anything starting with ``en_us`` / ``us``
        biases ambiguous numeric dates toward month-first; everything else
        (and the default) biases toward day-first, which is the global norm.

    Returns
    -------
    The ISO date string, or ``None`` if the input cannot be parsed.
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None

    prefer_day_first = True
    if locale_hint:
        h = locale_hint.strip().lower().replace("-", "_")
        if h.startswith("en_us") or h == "us" or h == "usa":
            prefer_day_first = False

    for parser in (_try_iso, _try_textual):
        result = parser(text)
        if result:
            return result
    return _try_numeric(text, prefer_day_first)


# --------------------------------------------------------------------------- #
# Amount / currency normalization
# --------------------------------------------------------------------------- #

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",   # euro
    "£": "GBP",   # pound
    "¥": "JPY",   # yen
    "₹": "INR",   # rupee
    "₩": "KRW",   # won
    "₽": "RUB",   # ruble
}

# ISO codes we recognise when written out explicitly.
CURRENCY_CODES = {
    "USD", "EUR", "GBP", "JPY", "INR", "CAD", "AUD", "CHF",
    "CNY", "KRW", "RUB", "SEK", "NOK", "DKK", "NZD", "SGD", "HKD",
}


@dataclass
class NormalizedAmount:
    """A parsed monetary amount.

    Attributes
    ----------
    value:
        The numeric value as a :class:`decimal.Decimal`.
    currency:
        ISO-4217 code if one could be inferred, else ``None``.
    """

    value: Decimal
    currency: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.currency:
            return f"{self.value} {self.currency}"
        return str(self.value)


def detect_currency(text: str) -> Optional[str]:
    """Infer an ISO-4217 currency code from symbols or codes in *text*."""
    if not text:
        return None
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    # Explicit 3-letter codes, e.g. "EUR 12,00" or "12.00 USD".
    for token in re.findall(r"[A-Za-z]{3}", text):
        up = token.upper()
        if up in CURRENCY_CODES:
            return up
    return None


_AMOUNT_BODY = re.compile(r"[0-9][0-9.,\s' ]*[0-9]|[0-9]")


def _clean_numeric_body(body: str) -> Optional[Decimal]:
    """Convert a numeric token (no currency) to a Decimal.

    Handles both ``1,234.56`` (Anglo) and ``1.234,56`` (European) by looking
    at the *last* separator: whichever of ``.`` or ``,`` appears last and is
    followed by 1-2 (or exactly 3-as-cents-edge) digits is treated as the
    decimal separator; the other is treated as a thousands grouping and
    stripped. When only one separator type is present we decide grouping vs
    decimal based on the digit run after it.
    """
    s = body.strip().replace(" ", "").replace(" ", "").replace("'", "")
    if not s:
        return None
    s = s.lstrip("+")
    negative = False
    if s.startswith("-"):
        negative = True
        s = s[1:]
    if not s:
        return None

    last_dot = s.rfind(".")
    last_comma = s.rfind(",")

    if last_dot == -1 and last_comma == -1:
        digits = s
    elif last_dot != -1 and last_comma != -1:
        # Both present: the one that appears last is the decimal separator.
        if last_dot > last_comma:
            # Anglo: comma = thousands, dot = decimal
            digits = s.replace(",", "")
        else:
            # European: dot = thousands, comma = decimal
            digits = s.replace(".", "").replace(",", ".")
    else:
        # Exactly one separator type present.
        sep = "." if last_dot != -1 else ","
        parts = s.split(sep)
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            # Looks like a decimal separator (e.g. 12,5 / 12.50).
            digits = parts[0] + "." + parts[1]
        elif all(len(p) == 3 for p in parts[1:]) and len(parts) > 1:
            # All trailing groups are 3 digits -> thousands grouping.
            digits = "".join(parts)
        elif len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            # Ambiguous like "1,234" / "1.234" -> treat as thousands grouping.
            digits = "".join(parts)
        else:
            digits = "".join(parts)

    if not re.fullmatch(r"\d+(?:\.\d+)?", digits):
        return None
    try:
        value = Decimal(digits)
    except InvalidOperation:
        return None
    return -value if negative else value


def normalize_amount(
    raw: str, default_currency: Optional[str] = None
) -> Optional[NormalizedAmount]:
    """Parse a raw monetary string into a :class:`NormalizedAmount`.

    Examples
    --------
    >>> normalize_amount("$1,234.56").value
    Decimal('1234.56')
    >>> normalize_amount("1.234,56 EUR").value
    Decimal('1234.56')
    >>> normalize_amount("EUR 1.234,56").currency
    'EUR'
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    currency = detect_currency(text) or default_currency

    m = _AMOUNT_BODY.search(text)
    if not m:
        return None
    body = m.group(0)
    # A leading minus that the body regex skipped (symbol before number).
    prefix = text[: m.start()]
    if "-" in prefix or text.strip().startswith("(") and text.strip().endswith(")"):
        body = "-" + body

    value = _clean_numeric_body(body)
    if value is None:
        return None
    return NormalizedAmount(value=value, currency=currency)
