# Doc Intelligence

Convert raw or OCR'd invoice / receipt text into structured JSON
(`vendor`, `invoice_number`, `date`, `currency`, `subtotal`, `tax`, `total`,
`line_items`) with a per-field **confidence** score.

The design philosophy: a **hard core implemented from scratch** — real
heuristics, a rule-based line-item state machine, hand-written date/amount
parsers and a transparent confidence model. Pretrained models (e.g. an OCR
engine) are only ever an *optional* front-end; everything in `core/` is pure
Python standard library.

## What I implemented from scratch

Everything in `core/` uses **only the Python standard library** — no
`dateutil`, no ML framework, no parsing libraries.

- **Layout-aware field extractor — [`core/fields.py`](core/fields.py).**
  Finds the vendor from the top-of-document block using company-suffix cues
  (Inc / LLC / Ltd / GmbH …), capitalization and address rejection; extracts
  the invoice number from labelled patterns; finds and normalizes the date;
  detects currency; and extracts `subtotal` / `tax` / `total` with an
  **"amount due" precedence rule** (an explicit *Amount Due* / *Total Due*
  overrides a plain *Total*). Each field records the *signals* that fired.

- **From-scratch line-item state machine — [`core/lineitems.py`](core/lineitems.py).**
  The centerpiece. A typed tokenizer splits each row into `WORD` / `NUMBER` /
  `MONEY` / `PERCENT` / `SEPARATOR` tokens; a row classifier determines the
  numeric-column *shape* (`single` / `double` / `triple` / `quad+`); and an
  **alignment state machine** maps the trailing numeric tokens to
  `(quantity, unit_price, line_total)` slots, using positional defaults and
  then an **arithmetic consistency check** (`qty * unit_price ≈ line_total`)
  to validate and, when needed, *repair* the alignment (re-ordering or
  re-selecting columns). It also rejects header rows, totals rows and postal
  address lines.

- **Confidence scoring — [`core/confidence.py`](core/confidence.py).**
  Each field's fired signals are combined with a **saturating sum**
  `score = 1 − ∏(1 − wᵢ)`. This is monotonic (more corroborating signals never
  lower the score), bounded in `[0, 1]`, and de-duplicates repeated signals.

- **Normalization layer — [`core/normalize.py`](core/normalize.py).**
  Dates → ISO-8601 via hand-written matchers (ISO, numeric `d/m/y` vs `m/d/y`
  with structural + locale disambiguation, textual `12 January 2024` /
  `Jan 12, 2024`, plus German month names), with a real leap-year / days-in-
  month validator. Amounts → `Decimal` + ISO-4217 currency, handling currency
  symbols, thousands separators and **both** the Anglo `1,234.56` and
  European `1.234,56` conventions, disambiguated structurally.

- **Extraction pipeline — [`core/pipeline.py`](core/pipeline.py).**
  Orchestrates the above into one call returning the structured JSON with a
  per-field and document-level confidence, including cross-field arithmetic
  corroboration (subtotal + tax ≈ total boosts the total's confidence).

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open <http://localhost:8000/> for the interactive HTML demo (paste an
invoice, see the JSON + confidence), or call the API directly:

```bash
curl -s http://localhost:8000/extract \
  -H 'content-type: application/json' \
  -d '{"text": "ACME LLC\nInvoice #: INV-1\nWidget  2  5.00  10.00\nTotal  $10.00", "locale_hint": "en_US"}'
```

With Docker:

```bash
docker build -t doc-intelligence .
docker run -p 8000:8000 doc-intelligence
```

> **Optional OCR front-end.** This project takes *text*. To process scanned
> images or PDFs, run an OCR pass first (e.g. `pytesseract`) and feed the
> resulting text to `/extract`. OCR is intentionally **not** a dependency —
> the from-scratch core is the point.

## API

### `POST /extract`

Request:

```json
{ "text": "<raw or OCR'd invoice text>", "locale_hint": "en_US" }
```

`locale_hint` is optional (e.g. `en_US`, `en_GB`, `de_DE`) and disambiguates
numeric dates / seeds a default currency.

Response:

```json
{
  "vendor": "ACME Web Solutions LLC",
  "invoice_number": "INV-2024-0099",
  "date": "2024-03-15",
  "currency": "USD",
  "subtotal": "2740.00",
  "tax": "232.90",
  "total": "2972.90",
  "line_items": [
    { "description": "Web design services", "quantity": "10",
      "unit_price": "150.00", "line_total": "1500.00",
      "currency": "USD", "arithmetic_ok": true, "confidence": 0.88 }
  ],
  "confidence": { "vendor": 0.76, "date": 0.75, "total": 0.95, "overall": 0.75 }
}
```

### `GET /health`

Liveness probe → `{"status": "ok", "version": "..."}`.

### `GET /`

Interactive HTML demo.

## Verify

Run the test suite (proves the core — date/amount normalization across both
separator conventions, line-item parsing + arithmetic validation, full-pipeline
extraction on each sample invoice, and confidence monotonicity):

```bash
pip install pytest
python -m pytest -q
```

You can also sanity-check the core with no server and no third-party deps:

```bash
python -c "from core.pipeline import extract_document; \
import json, pathlib; \
print(json.dumps(extract_document(pathlib.Path('samples/invoice_us.txt').read_text(), 'en_US').to_dict(), indent=2))"
```

See [EXPLAINER.md](EXPLAINER.md) for how the line-item state machine and the
confidence model work.
