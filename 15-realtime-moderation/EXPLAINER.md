# EXPLAINER — realtime-moderation

## What I implemented from scratch

A content-moderation pipeline whose two decision engines are both hand-built: a **multinomial Naive Bayes classifier** and a **policy rule engine**, fused into one explained verdict.

## Multinomial Naive Bayes (`core/naive_bayes.py`)

The classifier estimates, for a document `d` and class `c`:

```
P(c | d) ∝ P(c) · ∏_t P(t | c)^count(t,d)
```

Worked in **log space** to avoid floating-point underflow when multiplying many small probabilities:

```
log P(c | d) = log P(c) + Σ_t count(t,d) · log P(t | c)
```

The per-class word likelihoods use **Laplace (add-α) smoothing** so an unseen word doesn't zero out the whole product:

```
P(t | c) = (count(t,c) + α) / (total_count(c) + α·|V|)
```

Training is a single pass counting class frequencies and per-class word counts; prediction is the arg-max over classes of the log-posterior. Implemented with only `math` and the standard library — no NumPy, no scikit-learn — precisely so the mechanism is visible.

## Tokenizer (`core/tokenizer.py`)

Casefolds, strips, and splits text into tokens plus n-gram features. Deterministic and tested for stability so the classifier sees consistent inputs.

## Policy engine (`core/policy.py`, `core/rules.py`)

A small rule DSL: each rule names a category (toxicity, PII, spam, self-harm), a matcher (pattern or predicate), and a severity. Rules fire deterministically — useful for the things you must *never* get wrong (e.g. an exact credit-card pattern), where a probabilistic classifier is the wrong tool.

## Fusion, scoring, explanation (`core/scoring.py`, `core/explain.py`, `core/pipeline.py`)

The final verdict blends the rule hits with the classifier's posterior: rules contribute hard signals and categories, the classifier contributes a graded toxicity probability. Severity is aggregated, and `explain.py` assembles a human-readable trace — which rules matched, what the classifier scored — so a reviewer can see *why*, not just *what*. Explainability is a moderation requirement, not a nicety.

## Proof it works

`tests/` proves the Naive Bayes actually learns a separable toy distribution, the tokenizer round-trips and is stable, and the rules fire on crafted inputs.

## Limitations

- The bundled dataset is small and illustrative; production recall needs a larger labeled corpus.
- English-oriented; multilingual support would extend the tokenizer and retrain per language.
- Naive Bayes assumes feature independence — fine as a fast, explainable first line, but a transformer classifier would catch subtler context (and could slot in behind the same policy interface).
