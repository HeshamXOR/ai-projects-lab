# EXPLAINER — recsys-engine

## What I implemented from scratch

Three recommenders and a from-scratch ranking-evaluation harness, so the models can be compared honestly on held-out data.

## Matrix factorization (`core/mf.py`)

A low-rank latent-factor model. For user `u` and item `i`:

```
r̂(u,i) = μ + b_u[u] + b_i[i] + P[u] · Q[i]
```

- `μ` — global mean rating
- `b_u`, `b_i` — per-user and per-item bias terms (capture "this user rates high", "this item is loved")
- `P` (n_users × k), `Q` (n_items × k) — the latent factor matrices; their dot product is the personalized fit

Trained by **SGD** on the regularized squared error over *observed* ratings only:

```
J = Σ_(u,i)∈obs ( r_ui − r̂(u,i) )² + λ(‖P_u‖² + ‖Q_i‖² + b_u² + b_i²)
```

Each step samples an observed `(u,i)`, computes the error `e = r_ui − r̂`, and nudges every parameter along its gradient (e.g. `P_u += lr·(e·Q_i − λ·P_u)`). Regularization keeps the factors from overfitting the sparse observed entries.

## Item-item kNN (`core/itemknn.py`)

Classic neighborhood collaborative filtering: represent each item by its column of user interactions, score item similarity by cosine, and recommend items similar to those the user already liked. A strong, interpretable non-parametric baseline.

## Popularity baseline (`core/popularity.py`)

Recommends the globally most-popular items. This is the bar every personalized model must clear — if MF can't beat "just show everyone the hits," it isn't earning its complexity. Including it is honest engineering.

## Evaluation harness (`core/eval.py`)

Splits interactions into train/test, trains each model on train, and scores top-K recommendations against held-out test items with metrics implemented from scratch:

- **Precision@K** — fraction of the K recommendations that were relevant
- **Recall@K** — fraction of the user's relevant items that appeared in the top K
- **NDCG@K** — normalized discounted cumulative gain, which rewards putting relevant items *higher* in the ranking (a hit at rank 1 counts more than at rank 10)

## Proof it works

`tests/` proves the MF update equations are correct (loss decreases, shapes hold) and that **MF outperforms the popularity baseline** on the bundled sample by the ranking metrics — the result that justifies the model.

## Limitations

- The bundled MovieLens-style sample is small; production would use a full interaction log and a time-based split.
- Batch training only — no online/incremental factor updates.
- Implicit-feedback weighting (e.g. ALS with confidence) would be the next step for click/view data rather than explicit ratings.
