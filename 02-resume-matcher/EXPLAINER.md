# EXPLAINER — Resume Matcher: a real NLP pipeline

## What I implemented from scratch

- **TF-IDF vectorizer** with n-grams and L2 normalization (`core/tfidf.py`) — no scikit-learn.
- **Logistic regression** trained by gradient descent — forward pass, binary cross-entropy, gradients, and the update all written out (`core/logreg.py`).
- **Skill co-occurrence graph** that suggests transferable skills via graph traversal (`core/skillgraph.py`) — no networkx.

The pretrained sentence-embedding model is just *one* of three matching signals.

## How it works

**Three signals, blended:**
1. **Semantic similarity** — embedding cosine (the pretrained part). Captures paraphrase.
2. **TF-IDF cosine** (from scratch) — term-weighted overlap. `TF × IDF` downweights common words and rewards rare, meaningful ones; rows are L2-normalized so cosine is a dot product. Captures exact-term alignment the embeddings smooth over.
3. **Skill coverage** — fraction of the job's detected skills present in the resume.

**Skill graph for gap-closing:** edges connect skills that co-occur; given your skills, summing edge weights to non-possessed neighbors ranks the skills you're "closest" to — so the gap analysis suggests *what to learn next*, not just what's missing.

**Optional learned scorer:** `core/logreg.py` can replace the hand-picked blend weights with a logistic-regression model trained on labeled (resume, jd, match?) pairs — the from-scratch demonstration that the blend *could* be learned.

## Proof it works

`tests/test_core.py` (run `pytest`):
- TF-IDF L2-normalizes and correctly downweights common terms (lower IDF for "the" than "kubernetes").
- The sigmoid is numerically stable at ±1000.
- Logistic regression learns separable blobs to >98% accuracy with a decreasing loss curve.
- The skill graph points docker+python users toward kubernetes/aws.

## Limitations

- The skill vocabulary and co-occurrence corpus are small/curated; a production version would mine them from real postings.
- TF-IDF + logreg are deliberately classical — the point is to show the NLP fundamentals end to end, not to beat a large model.
