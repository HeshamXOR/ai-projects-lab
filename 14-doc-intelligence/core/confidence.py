"""Confidence scoring.

Each extracted field accumulates a set of *signals* during extraction (see
:mod:`core.fields` and :mod:`core.lineitems`). This module turns those signals
into a calibrated confidence score in ``[0, 1]``.

The model is deliberately simple and transparent (no learned weights): every
signal has a hand-assigned weight reflecting how much it corroborates the
extraction, and the score is a saturating sum

    score = 1 - prod(1 - w_i)

over the fired signals. This has three nice properties the tests rely on:

* **Monotonicity** -- adding more corroborating signals never lowers the score.
* **Saturation** -- the score approaches but never exceeds 1.
* **Bounded** -- a single weak signal yields a small score; multiple strong
  signals push toward high confidence.

Arithmetic consistency on line items is folded in as an extra signal so that a
row whose ``qty*unit_price`` matches its total is scored higher than one that
does not.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

__all__ = [
    "SIGNAL_WEIGHTS",
    "score_signals",
    "score_field",
    "score_line_item",
    "aggregate_confidence",
]


# Per-signal weights. Higher = stronger corroboration. These are intentionally
# < 1 so that the saturating-sum combiner stays below 1 until several signals
# agree.
SIGNAL_WEIGHTS: Dict[str, float] = {
    # generic
    "label_keyword_present": 0.55,
    "format_matched": 0.45,
    "contains_digits": 0.25,
    # vendor
    "company_suffix_present": 0.6,
    "in_top_block": 0.4,
    "all_caps_heading": 0.3,
    "title_case_heading": 0.25,
    "first_line_fallback": 0.1,
    # invoice number
    "hash_number_pattern": 0.4,
    # currency
    "currency_detected": 0.4,
    "explicit_iso_code": 0.45,
    "symbol_present": 0.35,
    # totals
    "amount_due_precedence": 0.5,
    # line items
    "arithmetic_consistent": 0.6,
    "has_quantity": 0.2,
    "has_unit_price": 0.2,
    "has_currency": 0.15,
}

# Default weight for any unknown signal so the model degrades gracefully.
_DEFAULT_WEIGHT = 0.2


def score_signals(signals: Iterable[str]) -> float:
    """Combine signals into a saturating-sum confidence in ``[0, 1]``.

    ``score = 1 - prod(1 - w_i)``. Duplicate signals are de-duplicated so a
    repeated signal cannot inflate the score.
    """
    seen = set()
    product = 1.0
    for sig in signals:
        if sig in seen:
            continue
        seen.add(sig)
        w = SIGNAL_WEIGHTS.get(sig, _DEFAULT_WEIGHT)
        # Clamp weight defensively.
        w = max(0.0, min(0.99, w))
        product *= (1.0 - w)
    score = 1.0 - product
    return round(score, 4)


def score_field(signals: Iterable[str]) -> float:
    """Confidence for a scalar header field given its fired signals."""
    sigs = list(signals)
    if not sigs:
        return 0.0
    return score_signals(sigs)


def score_line_item(
    *,
    arithmetic_ok: bool,
    has_quantity: bool,
    has_unit_price: bool,
    has_currency: bool,
    extra_signals: Iterable[str] = (),
) -> float:
    """Confidence for a single parsed line item."""
    signals: List[str] = list(extra_signals)
    if arithmetic_ok:
        signals.append("arithmetic_consistent")
    if has_quantity:
        signals.append("has_quantity")
    if has_unit_price:
        signals.append("has_unit_price")
    if has_currency:
        signals.append("has_currency")
    # A line item with nothing but a total still has a base format match.
    signals.append("format_matched")
    return score_signals(signals)


def aggregate_confidence(field_scores: Dict[str, float]) -> float:
    """Overall document confidence as the mean of present field scores."""
    present = [v for v in field_scores.values() if v is not None]
    if not present:
        return 0.0
    return round(sum(present) / len(present), 4)
