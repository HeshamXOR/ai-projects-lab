"""End-to-end pipeline tests over the realistic sample invoices."""

from __future__ import annotations

from decimal import Decimal

from core.pipeline import extract_document


class TestUSInvoice:
    def test_fields(self, sample_us):
        r = extract_document(sample_us, "en_US")
        assert r.vendor == "ACME Web Solutions LLC"
        assert r.invoice_number == "INV-2024-0099"
        assert r.date == "2024-03-15"
        assert r.currency == "USD"
        assert r.total == "2972.90"          # amount-due precedence
        assert r.subtotal == "2740.00"
        assert r.tax == "232.90"

    def test_line_items(self, sample_us):
        r = extract_document(sample_us, "en_US")
        assert len(r.line_items) == 3
        descs = [li["description"] for li in r.line_items]
        assert any("Web design" in d for d in descs)
        assert all(li["arithmetic_ok"] for li in r.line_items)

    def test_amount_due_beats_total(self, sample_us):
        # The document has both a (none) and an Amount Due; total must be the
        # amount due (2972.90), not the subtotal.
        r = extract_document(sample_us, "en_US")
        assert Decimal(r.total) == Decimal("2972.90")


class TestEUInvoice:
    def test_fields(self, sample_eu):
        r = extract_document(sample_eu, "de_DE")
        assert "Müller Elektronik GmbH" in r.vendor
        assert r.currency == "EUR"
        assert r.date == "2024-02-14"
        assert r.total == "1729.69"
        assert r.subtotal == "1453.52"

    def test_european_line_item_amounts(self, sample_eu):
        r = extract_document(sample_eu, "de_DE")
        totals = [li["line_total"] for li in r.line_items]
        assert "1234.56" in totals     # 1.234,56 normalized
        # Address line must not appear as an item.
        assert not any("straße" in li["description"].lower() for li in r.line_items)


class TestUKInvoice:
    def test_fields(self, sample_uk):
        r = extract_document(sample_uk, "en_GB")
        assert "Brightwave Consulting Ltd" in r.vendor
        assert r.currency == "GBP"
        assert r.invoice_number == "BW-7781"
        assert r.date == "2024-01-12"        # "Jan 12, 2024"
        assert r.total == "2772.00"          # "Total Due" precedence

    def test_line_items(self, sample_uk):
        r = extract_document(sample_uk, "en_GB")
        assert len(r.line_items) == 3
        assert all(li["arithmetic_ok"] for li in r.line_items)


class TestReceipt:
    def test_fields(self, sample_receipt):
        r = extract_document(sample_receipt)
        assert "CORNER MARKET" in r.vendor
        assert r.currency == "USD"
        assert r.total == "21.30"
        assert r.date == "2024-01-22"

    def test_all_four_items(self, sample_receipt):
        r = extract_document(sample_receipt)
        assert len(r.line_items) == 4
        descs = " ".join(li["description"].lower() for li in r.line_items)
        for word in ("banana", "milk", "bread", "egg"):
            assert word in descs


class TestConfidenceContract:
    def test_overall_present(self, sample_us):
        r = extract_document(sample_us, "en_US")
        assert "overall" in r.confidence
        assert 0.0 <= r.confidence["overall"] <= 1.0

    def test_every_extracted_field_has_confidence(self, sample_us):
        r = extract_document(sample_us, "en_US")
        for key in ("vendor", "invoice_number", "date", "total"):
            assert key in r.confidence
            assert r.confidence[key] > 0.0

    def test_empty_input_is_safe(self):
        r = extract_document("")
        assert r.line_items == []
        assert r.confidence.get("overall", 0.0) == 0.0

    def test_to_dict_shape(self, sample_us):
        d = extract_document(sample_us, "en_US").to_dict()
        for key in ("vendor", "invoice_number", "date", "currency",
                    "subtotal", "tax", "total", "line_items", "confidence"):
            assert key in d
