# recsys-engine

A recommendation engine with a real **offline evaluation harness** — matrix factorization, item-item kNN, and a popularity baseline, all benchmarked against each other on held-out data.

## What I implemented from scratch

- **Matrix factorization** by SGD (`core/mf.py`) — a latent-factor model `r̂(u,i) = μ + b_u + b_i + P_u · Q_i`, trained by stochastic gradient descent on the regularized squared error over observed ratings. Pure NumPy.
- **Item-item kNN** (`core/itemknn.py`) — cosine-similarity collaborative filtering.
- **A popularity baseline** (`core/popularity.py`) — the honest reference every recommender must beat.
- **The evaluation harness** (`core/eval.py`) — Precision@K, Recall@K, and **NDCG@K** computed from scratch, with a train/test split, so model quality is measured, not asserted.

## What it does

`POST /recommend` returns top-N items for a user from the trained model. The point of the project is the **eval harness**: it trains all three models on a bundled MovieLens-style sample and reports ranking metrics, demonstrating that the from-scratch matrix factorization beats the popularity baseline on held-out interactions.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload          # http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t recsys-engine . && docker run -p 8000:8000 recsys-engine
```

## API

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `POST` | `/recommend` | `{ "user_id": 1, "k": 10 }` | ranked item list with scores |
| `GET`  | `/health`    | — | service status |

## Verify

```bash
pytest -q     # proves the MF math and that MF beats the popularity baseline on the sample
```

## How it works

See [EXPLAINER.md](EXPLAINER.md) for the latent-factor model, the SGD update equations, and how the ranking metrics are computed.

## Limitations

- Bundled sample is small; real deployments would train on a full interaction log.
- Batch-trained (no online updates); a production system would add incremental factor updates.
