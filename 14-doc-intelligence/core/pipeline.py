"""Extraction pipeline.

Orchestrates the from-scratch components into a single call:

    raw text -> {vendor, invoice_number, date, currency, subtotal, tax,
                 total, line_items, confidence}

The pipeline is pure-stdlib and deterministic. It wires the field extractors
(:mod:`core.fields`), the line-item state machine (:mod:`core.lineitems`), the
normalizers (:mod:`core.normalize`) and the confidence model
(:mod:`core.confidence`) together, attaching a per-field confidence score and a
document-level aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from . import confidence as conf
from . import fields as F
from .lineitems import LineItem, parse_line_items
from .normalize import NormalizedAmount

__all__ = ["ExtractionResult", "extract_document"]


def _amount_to_str(amt: Optional[NormalizedAmount]) -> Optional[str]:
    if amt is None:
        return None
    return str(amt.value)


@dataclass
class ExtractionResult:
    vendor: Optional[str] = None
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[str] = None
    tax: Optional[str] = None
    total: Optional[str] = None
    line_items: List[Dict[str, Any]] = dc_field(default_factory=list)
    confidence: Dict[str, float] = dc_field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "invoice_number": self.invoice_number,
            "date": self.date,
            "currency": self.currency,
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total": self.total,
            "line_items": self.line_items,
            "confidence": self.confidence,
        }


def extract_document(text: str, locale_hint: Optional[str] = None) -> ExtractionResult:
    """Run the full extraction pipeline over *text*.

    Parameters
    ----------
    text:
        Raw or OCR'd invoice / receipt text. May contain positional hints such
        as line breaks and column spacing, which the extractors exploit.
    locale_hint:
        Optional locale string (e.g. ``"en_US"``, ``"de_DE"``). Used to
        disambiguate numeric dates and to seed a default currency.

    Returns
    -------
    :class:`ExtractionResult`
    """
    if text is None:
        text = ""

    result = ExtractionResult()
    field_conf: Dict[str, float] = {}

    # --- currency first so it can seed amount parsing ---------------------- #
    cur_ext = F.extract_currency(text)
    default_currency = cur_ext.value
    if cur_ext.value:
        result.currency = cur_ext.value
        field_conf["currency"] = conf.score_field(cur_ext.signals)

    # --- vendor ------------------------------------------------------------ #
    vendor_ext = F.extract_vendor(text)
    if vendor_ext.value:
        result.vendor = str(vendor_ext.value)
        field_conf["vendor"] = conf.score_field(vendor_ext.signals)

    # --- invoice number ---------------------------------------------------- #
    inv_ext = F.extract_invoice_number(text)
    if inv_ext.value:
        result.invoice_number = str(inv_ext.value)
        field_conf["invoice_number"] = conf.score_field(inv_ext.signals)

    # --- date -------------------------------------------------------------- #
    date_ext = F.extract_date(text, locale_hint)
    if date_ext.value:
        result.date = str(date_ext.value)
        field_conf["date"] = conf.score_field(date_ext.signals)

    # --- totals ------------------------------------------------------------ #
    totals = F.extract_totals(text, default_currency)
    if totals.subtotal is not None:
        result.subtotal = _amount_to_str(totals.subtotal.value)
        field_conf["subtotal"] = conf.score_field(totals.subtotal.signals)
    if totals.tax is not None:
        result.tax = _amount_to_str(totals.tax.value)
        field_conf["tax"] = conf.score_field(totals.tax.signals)
    if totals.total is not None:
        result.total = _amount_to_str(totals.total.value)
        field_conf["total"] = conf.score_field(totals.total.signals)
        # If we never detected a currency but the total carried one, adopt it.
        if not result.currency and isinstance(totals.total.value, NormalizedAmount):
            if totals.total.value.currency:
                result.currency = totals.total.value.currency

    # --- line items -------------------------------------------------------- #
    items: List[LineItem] = parse_line_items(text)
    item_dicts: List[Dict[str, Any]] = []
    item_scores: List[float] = []
    for it in items:
        score = conf.score_line_item(
            arithmetic_ok=it.arithmetic_ok,
            has_quantity=it.quantity is not None,
            has_unit_price=it.unit_price is not None,
            has_currency=bool(it.currency or result.currency),
        )
        d = it.to_dict()
        if d.get("currency") is None and result.currency:
            d["currency"] = result.currency
        d["confidence"] = score
        item_dicts.append(d)
        item_scores.append(score)

    result.line_items = item_dicts
    if item_scores:
        field_conf["line_items"] = round(sum(item_scores) / len(item_scores), 4)

    # --- cross-field arithmetic corroboration ------------------------------ #
    # If subtotal + tax approx total, boost the total's confidence.
    _maybe_boost_total(result, totals, field_conf)

    field_conf["overall"] = conf.aggregate_confidence(
        {k: v for k, v in field_conf.items() if k != "overall"}
    )
    result.confidence = field_conf
    return result


def _maybe_boost_total(
    result: ExtractionResult, totals: F.Totals, field_conf: Dict[str, float]
) -> None:
    try:
        if not (totals.subtotal and totals.tax and totals.total):
            return
        sub = totals.subtotal.value.value
        tax = totals.tax.value.value
        tot = totals.total.value.value
        if abs((sub + tax) - tot) <= max(Decimal("0.02"), tot * Decimal("0.01")):
            # Re-score the total with an extra corroborating signal.
            sigs = list(totals.total.signals) + ["arithmetic_consistent"]
            field_conf["total"] = conf.score_field(sigs)
    except Exception:
        # Confidence boosting must never break extraction.
        return
