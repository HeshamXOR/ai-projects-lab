# EXPLAINER — how privacy-pii-vault works

This document explains the algorithms behind the four from-scratch components:
detection rules, the Luhn checksum, HMAC format-preserving tokenization, and
k-anonymity. Everything runs on the Python standard library.

---

## 1. PII detection (`core/detect.py`, `core/gazetteer.py`)

PII splits into two structurally different classes, handled differently.

### 1a. Structured PII — regex + validators

Email, phone, SSN, credit card, IPv4, and dates have a recognizable surface
form, so each gets a bounded regular expression. Crucially, a regex match is
only a *candidate*; a **validator** confirms it. This two-stage design is what
gives high precision:

- **Email** — pragmatic RFC-5322 subset: `local@domain.tld`, TLD 2–24 chars.
- **Phone** — optional `+country`, optional `(area)` parens, separators of
  space/dot/dash. A post-filter requires ≥ 10 digits so bare years/IDs do not
  match.
- **SSN** — `NNN-NN-NNNN` with negative lookaheads rejecting SSA-invalid area
  numbers (`000`, `666`, `900–999`), group `00`, and serial `0000`.
- **Credit card** — loose 13–19 digit run with optional grouping; **every
  candidate must pass Luhn** (§2), so non-card digit runs are discarded.
- **IPv4** — loose dotted-quad; `_valid_ipv4` enforces 4 octets, each 0–255,
  and rejects ambiguous leading zeros.
- **Date** — ISO `YYYY-MM-DD`, US `M/D/YYYY`, and `Month DD, YYYY`.

Each detection carries a confidence (cards/SSN/email ≈ 0.97–0.99; phone/date a
bit lower because the patterns are looser).

### 1b. Unstructured PII — gazetteer + context cues

Names and organizations are an *open class* (no fixed shape), so we score
capitalized-token runs using evidence:

1. **Tokenize** with character offsets.
2. **Find maximal runs** of capitalized tokens (`John Smith`,
   `Acme Technologies`).
3. **Score** the run:
   - org keyword inside or trailing the run (`Inc`, `LLC`, `Technologies`) ⇒
     **ORG** (and the span extends to absorb a trailing `Inc`);
   - a preceding honorific (`Dr.`, `Mr.`) ⇒ **PERSON**, even for names not in
     the gazetteer (handles novel surnames);
   - gazetteer hits on given-name/surname tokens ⇒ **PERSON**, strongest when
     both a given name and a surname appear;
   - a preceding trigger phrase (`named`, `works at`) nudges type/confidence.
4. Multi-word **known orgs** (e.g. "World Health Organization") are matched
   directly with word-boundary checks.

### 1c. Overlap resolution

Because the policy engine edits the string by character span, overlapping
detections would corrupt offsets. `_resolve_overlaps` sorts by
`(confidence desc, length desc)` and greedily keeps a detection only if it does
not intersect an already-kept one, then re-sorts into document order. Result:
a clean, non-overlapping span set.

---

## 2. The Luhn checksum (`luhn_is_valid`)

The Luhn (mod-10) algorithm is a check-digit formula that catches all
single-digit errors and most adjacent transpositions. Walking the digits
**right to left**:

1. Keep every first digit as-is.
2. **Double every second digit**; if the result exceeds 9, subtract 9
   (equivalent to adding the two decimal digits, e.g. `8×2=16 → 1+6 = 7`).
3. Sum everything. The number is valid iff the total is a multiple of 10.

```
4 2 4 2 ...               (digits)
^   ^                      doubled positions (every 2nd from the right)
```

We require ≥ 12 digits so short numeric runs cannot masquerade as cards, and we
ignore spaces/dashes so grouped forms like `4242 4242 4242 4242` validate.

The tokenizer reuses this idea in reverse: after generating fake card digits it
**solves for the final check digit** so the surrogate also passes Luhn.

---

## 3. HMAC format-preserving tokenization (`core/tokenize.py`)

### Goal
Replace a value with a fake-but-realistic surrogate that (a) keeps the input's
format, (b) is deterministic under a secret key, and (c) is reversible via a
vault.

### Determinism via HMAC
A token's randomness comes from
`HMAC-SHA256(key, "TYPE\x00value")`. HMAC is a keyed pseudo-random function: its
output is uniformly unpredictable *without* the key, but fully reproducible
*with* it. Mixing the PII **type** into the message ensures the same string
tokenizes differently across types (no cross-type collisions).

To emit arbitrarily many digits/letters deterministically we run an
**HMAC-DRBG-style stream**: re-HMAC with an incrementing counter
(`"TYPE:0"`, `"TYPE:1"`, …) and map each output byte to a digit (`byte % 10`) or
letter (`byte % 26`).

### Format preservation per type
- **Card** — replace each digit with a derived digit, preserve separator
  positions, then fix the last digit so the whole surrogate passes Luhn ⇒ a
  16-digit card → 16-digit Luhn-valid token.
- **Phone / SSN / IP** — replace digits in place, keep punctuation/parens; IP
  octets are taken `byte % 256` so each stays 0–255.
- **Email** — derive a new local part and domain label of the same lengths,
  **preserve the `@` and the original TLD** ⇒ still email-shaped.
- **Person / Org** — replace each word with a derived alphabetic word
  (capitalized); org keywords (`Inc`) are kept so it still reads like a company.

### Reversibility: the Vault
The deterministic token is a *pseudonym*; the authoritative reverse index is
the `Vault`, a thread-safe map keyed by `(type, token) → original`.
`tokenize()` is idempotent (same value → cached token, no duplicate entry);
`detokenize()` returns the original or `None`. The vault is what makes the
otherwise one-way pseudonym recoverable.

### Key handling
`derive_key` stretches a passphrase with **PBKDF2-HMAC-SHA256** (200k
iterations) into a 32-byte key, making brute force materially harder than a
single hash.

### Security note
The vault stores plaintext originals — see the README caveat. HMAC determinism
intentionally reveals input equality to key holders (referential integrity)
but is a linkage vector across joined datasets.

---

## 4. k-anonymity (`core/risk.py`)

### The threat
Direct identifiers can be removed yet individuals re-identified by combining
**quasi-identifiers** (QIs) like ZIP + birth date + sex and linking to public
data. (Sweeney: ~87% of the US population is unique on those three.)

### The definition
A dataset is **k-anonymous** on a QI set if every record is indistinguishable
from at least `k − 1` others on those QIs. Equivalently:

1. Group rows into **equivalence classes** by their QI tuple.
2. `k` = the size of the **smallest** equivalence class.
3. The table is k-anonymous iff `k ≥ threshold`.
4. Rows in any class smaller than the threshold are **re-identifiable** and get
   flagged for suppression/generalization.

Our implementation builds the classes in one pass (`build_equivalence_classes`),
takes the min size, and collects the indices of rows in undersized classes.

### l-diversity (bonus)
k-anonymity can still leak a **sensitive** attribute if a class is homogeneous
(everyone in a class shares the same diagnosis). **l-diversity** requires each
class to contain at least `l` distinct sensitive values. We report, per
sensitive attribute, the *minimum* distinct-value count across classes — if
that `l` is 1, at least one class leaks the value despite being k-anonymous.

### Worked example (see `tests/test_risk.py`)
Six rows over `{zip, age, sex}` form three classes of sizes `{3, 2, 1}`, so
`k = 1`; the single unique row (index 5) is flagged. The F-class shares one
disease, so `l = 1` for `disease` — a homogeneity leak the scorer surfaces.
