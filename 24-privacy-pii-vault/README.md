# privacy-pii-vault

A PII detection, redaction, reversible tokenization, and re-identification-risk
service. The detection engine, format-preserving tokenizer, Luhn checksum, and
k-anonymity scorer are all built from scratch on the Python standard library --
no spaCy, no presidio, no pandas.

## What I implemented from scratch

- **PII detector** (`core/detect.py`) — regex + validators for structured PII
  (email, phone, SSN, credit card, IPv4, dates) plus a gazetteer + context-cue
  NER for PERSON / ORG. Credit cards are confirmed with a hand-written **Luhn
  checksum**; IPs are octet-range validated; SSNs reject SSA-invalid ranges.
  Overlapping detections are resolved by a confidence/length greedy pass so
  spans never collide. Each hit is a `Detection(type, start, end, value,
  confidence)`.
- **Name/Org gazetteer** (`core/gazetteer.py`) — frozensets of given names,
  surnames, org keywords (`Inc`, `LLC`, ...), honorifics (`Dr.`), and trigger
  phrases (`works at`, `named`) that drive the open-class NER.
- **Format-preserving reversible tokenization** (`core/tokenize.py`) —
  deterministic surrogates derived from **HMAC-SHA256(key, type‖value)** via an
  HMAC-DRBG-style digit/letter stream. A 16-digit card maps to a 16-digit token
  that *also passes Luhn*; an email stays email-shaped (TLD preserved); phones,
  SSNs, and IPs keep their layout. A keyed `Vault` stores the reverse mapping so
  `detokenize()` restores the original. Same key+input ⇒ same token; different
  key ⇒ different token. Key derived with PBKDF2-HMAC-SHA256.
- **Policy engine** (`core/policy.py`) — per-type actions REDACT / MASK (keep
  last 4) / TOKENIZE / HASH / ALLOW, applied right-to-left so replacements of
  differing length never corrupt earlier spans.
- **k-anonymity risk scorer** (`core/risk.py`) — builds equivalence classes over
  quasi-identifier columns, computes `k` = smallest class size, flags rows below
  a threshold as re-identifiable, plus a simple **l-diversity** note per
  sensitive attribute.

## Run it

```bash
pip install -r requirements.txt

# Secrets (use a real secrets manager in production):
export VAULT_PASSPHRASE="my-strong-passphrase"
export VAULT_API_KEY="my-detokenize-key"

uvicorn app:app --reload          # http://127.0.0.1:8000  (docs at /docs)
```

Run the tests (they prove the core):

```bash
pytest -q
```

Docker:

```bash
docker build -t pii-vault .
docker run -p 8000:8000 -e VAULT_PASSPHRASE=... -e VAULT_API_KEY=... pii-vault
```

## API

### `GET /health`
Liveness + vault entry count.

### `POST /redact`
Detect PII and apply a policy.

```json
{
  "text": "Email alice.johnson@example.com, card 4242 4242 4242 4242.",
  "default_action": "REDACT",
  "policy": { "EMAIL": "TOKENIZE", "CREDIT_CARD": "MASK" },
  "mask_keep": 4,
  "min_confidence": 0.5
}
```
Returns `redacted_text`, every `detection` (type/span/value/confidence), and any
`issued_tokens` (for reversible TOKENIZE entries).

### `POST /detokenize`  *(requires `X-API-Key` header)*
```json
{ "tokens": [ { "type": "EMAIL", "token": "qkft@xbnt.com" } ] }
```
Returns each token's restored `original` (or `found: false`). Gated behind the
API key because re-exposing raw PII is the privileged operation.

### `POST /risk-score`
```json
{
  "rows": [ {"zip":"10001","age":30,"sex":"M"}, ... ],
  "quasi_identifiers": ["zip","age","sex"],
  "threshold": 2,
  "sensitive_attrs": ["disease"]
}
```
Returns `k`, `is_k_anonymous`, `flagged_rows`, `class_sizes`, and `l_diversity`.

## Security caveat

The `Vault` stores **plaintext originals in memory**, keyed by surrogate token.
This is a demonstration store. In production: persist the vault in an
access-controlled, encrypted-at-rest, HSM-backed system; inject
`VAULT_PASSPHRASE` / `VAULT_API_KEY` from a secrets manager; and run a single
worker (or externalize the vault) so mappings are shared. HMAC determinism means
token equality reveals input equality to a key holder — intentional for
referential integrity, but a known linkage vector. Combine tokenized releases
with the k-anonymity scorer before publishing.
