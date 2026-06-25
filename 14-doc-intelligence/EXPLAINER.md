# Explainer: the line-item state machine & the confidence model

This document explains the two pieces of from-scratch engineering that matter
most: how invoice **line items** are parsed by a rule-based state machine, and
how each extracted field gets a **confidence** score.

---

## 1. The line-item state machine (`core/lineitems.py`)

Invoice line-item tables are deceptively hard: the same logical table appears
with wildly different spacing, column counts and conventions, often after OCR
has mangled the alignment. Rather than reach for a model, we parse each row
with a small, inspectable pipeline.

### Stage 1 — Tokenization

Each physical line is split into **columns** first (tabs, pipes, or runs of two
or more spaces — the visual column delimiters invoices actually use), then each
column into space-separated tokens. Every token is classified into a type:

| Type        | Example         | Meaning                              |
|-------------|-----------------|--------------------------------------|
| `MONEY`     | `$1,000.00`     | numeric **with** a currency symbol/code |
| `NUMBER`    | `10`, `150.00`  | bare numeric (could be qty or price) |
| `PERCENT`   | `8.5%`          | a percentage (usually tax/discount)  |
| `WORD`      | `Consulting`    | part of the description              |
| `SEPARATOR` | column gap      | preserves column structure           |

Numeric tokens are parsed through the same `normalize_amount` used everywhere,
so `1.234,56` and `$1,234.56` both become `Decimal('1234.56')` at tokenization
time.

### Stage 2 — Is this even an item row?

Before alignment we reject rows that are *not* items:

- **Totals / metadata** rows — any line containing `subtotal`, `tax`, `total`,
  `amount due`, `invoice`, `date`, … (English **and** German aliases).
- **Header** rows — lines where every word is a column header
  (`description`, `qty`, `unit`, `price`, `amount`, `Beschreibung`, `Menge`, …).
- **Address** rows — detected with word-boundary keyword matching (`street`,
  `ave`, `suite`, …), a US `STATE + ZIP` pattern, a German `…straße 17`
  pattern, or a leading street number with no decimal price column. Word
  boundaries matter: a naive substring check would reject *"Sou**rd**ough"*
  because it contains `rd`.

A surviving candidate must have at least one numeric token **and** at least one
descriptive word.

### Stage 3 — Column-shape classification

We take the numeric tokens that trail the last descriptive word and count them:

- `quad+` — 4+ numeric columns (`desc qty unit (tax) amount`)
- `triple` — 3 (`desc qty unit total`)
- `double` — 2 (`desc qty amount` **or** `desc unit amount`)
- `single` — 1 (`desc amount`)

### Stage 4 — Alignment with arithmetic repair (the state machine)

This is the core. Given the shape and the trailing numeric values, we assign
`(quantity, unit_price, line_total)` using positional defaults, then validate
with the invariant

```
quantity × unit_price ≈ line_total      (within 1 cent or 2% relative)
```

The arithmetic check is what turns a guess into a decision:

- **`triple`** — default to `(qty, unit, total)`. If the product check fails,
  the machine **brute-forces a small set of column permutations** and adopts
  the first that satisfies the invariant (recovering rows printed as
  `qty total unit`, etc.).
- **`quad+`** — take the first number as quantity and the last as total; if
  `qty × unit` doesn't match, **re-select the unit price** from the
  second-to-last column (handles an interposed tax/discount column).
- **`double`** — if the first value is a small whole number it's treated as a
  quantity and the unit price is **derived** as `amount / qty`; otherwise it's
  treated as a unit price with implicit quantity 1.
- **`single`** — quantity 1, `unit_price = line_total`.

Every row carries an `arithmetic_ok` flag recording whether the invariant held.
A row whose math is consistent is later scored higher by the confidence model.
Rows that fail the check are **still returned** (with `arithmetic_ok = False`)
rather than dropped — extraction degrades gracefully instead of silently losing
data.

---

## 2. The confidence model (`core/confidence.py`)

Every extractor records the **signals** that fired while producing a value —
for example `label_keyword_present`, `format_matched`, `company_suffix_present`,
`amount_due_precedence`, `arithmetic_consistent`. Confidence is a function of
which signals corroborate the extraction.

### The combiner

Each signal has a hand-assigned weight `wᵢ ∈ (0, 1)` reflecting how strongly it
corroborates a value. The signals are combined with a **saturating sum**:

```
score = 1 − ∏ (1 − wᵢ)
```

Reading this as probability-of-correct under independent evidence: each signal
independently "rules out" a fraction `(1 − wᵢ)` of the doubt, and the product
is the residual doubt. This gives three properties the tests assert directly:

1. **Monotonicity** — adding any corroborating signal can only *increase* the
   score (`s₁ < s₂ < s₃` as signals accumulate). This is the property a
   downstream consumer relies on to threshold low-confidence fields.
2. **Saturation** — the score asymptotically approaches but never reaches 1, so
   no finite set of heuristics ever claims certainty.
3. **De-duplication** — repeated signals are ignored, so a noisy extractor that
   emits the same signal twice cannot inflate its own score.

Unknown signals fall back to a small default weight, so the model degrades
gracefully if a new signal is added to an extractor before being weighted.

### Field vs. document confidence

- **Per field** — `score_field(signals)` over that field's signals.
- **Per line item** — `score_line_item(...)` folds in `arithmetic_consistent`,
  `has_quantity`, `has_unit_price`, `has_currency`.
- **Cross-field corroboration** — in the pipeline, if `subtotal + tax ≈ total`,
  the total gains an extra `arithmetic_consistent` signal and is re-scored.
- **Document level** — `overall` is the mean of all present field scores.

The result is a confidence map that is fully explainable: every number traces
back to a concrete list of signals that fired during extraction.
