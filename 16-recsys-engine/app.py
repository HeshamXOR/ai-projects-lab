"""FastAPI service exposing the from-scratch recommendation engine.

Endpoints
---------
- ``GET  /health``        -- liveness + model/catalogue summary.
- ``GET  /movies``        -- the movie catalogue (id, title, genre).
- ``POST /recommend``     -- ranked recommendations for a user from a chosen
                             model ("mf" | "popularity" | "itemknn").

All three models are trained at startup on the bundled dataset (fast on the
tiny data). Unknown users fall back to the popularity model, since the
personalised models have no factors for an unseen id.

The root path serves a tiny self-contained HTML demo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.data import Dataset, ensure_dataset
from core.itemknn import ItemKNN
from core.mf import MatrixFactorization
from core.popularity import PopularityRecommender

ModelName = Literal["mf", "popularity", "itemknn"]


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #
class RecommendRequest(BaseModel):
    """Request body for ``POST /recommend``."""

    user_id: int = Field(..., description="External user id to recommend for.")
    k: int = Field(10, ge=1, le=100, description="Number of items to return.")
    model: ModelName = Field("mf", description="Which recommender to use.")


class Recommendation(BaseModel):
    """A single ranked recommendation."""

    item_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    """Response body for ``POST /recommend``."""

    user_id: int
    model: str
    fallback: bool = Field(
        False, description="True if the request fell back to popularity."
    )
    recommendations: List[Recommendation]


class Movie(BaseModel):
    """A catalogue entry."""

    item_id: int
    title: str
    genre: str


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    n_users: int
    n_items: int
    n_ratings: int
    models: List[str]


# --------------------------------------------------------------------------- #
# Engine -- trains + holds all three models
# --------------------------------------------------------------------------- #
class Engine:
    """Holds the dataset and the three fitted models.

    Built once at startup. Provides a uniform :meth:`recommend` that maps
    external ids <-> internal indices and handles unknown-user fallback.
    """

    def __init__(self) -> None:
        self.dataset: Optional[Dataset] = None
        self.mf: Optional[MatrixFactorization] = None
        self.pop: Optional[PopularityRecommender] = None
        self.knn: Optional[ItemKNN] = None

    def train(self) -> None:
        """Load the bundled dataset and fit all three models."""
        ds = ensure_dataset()
        self.dataset = ds
        ratings = ds.ratings

        self.mf = MatrixFactorization(
            n_factors=8, n_epochs=40, lr=0.02, reg=0.05, seed=0
        ).fit(ratings, ds.n_users, ds.n_items)
        self.pop = PopularityRecommender(shrinkage=5.0).fit(ratings, ds.n_items)
        self.knn = ItemKNN(k_neighbors=20).fit(ratings, ds.n_users, ds.n_items)

    # ------------------------------------------------------------------ #
    def _model(self, name: ModelName):
        return {"mf": self.mf, "popularity": self.pop, "itemknn": self.knn}[name]

    def recommend(
        self, user_id: int, k: int, model: ModelName
    ) -> RecommendResponse:
        """Produce ranked recommendations, handling id mapping + fallback."""
        assert self.dataset is not None and self.pop is not None
        ds = self.dataset

        known = user_id in ds.user_index
        fallback = False
        chosen: ModelName = model

        # Personalised models cannot serve unknown users -> fall back.
        if not known and model in ("mf", "itemknn"):
            chosen = "popularity"
            fallback = True

        recommender = self._model(chosen)
        if recommender is None:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="model not trained")

        if known:
            u_idx = ds.user_index[user_id]
            exclude = ds.known_items_for_user(u_idx)
        else:
            u_idx = 0  # ignored by popularity
            exclude = np.empty(0, dtype=int)

        pairs = recommender.recommend(u_idx, k=k, exclude=exclude)

        recs: List[Recommendation] = []
        for item_idx, score in pairs:
            item_id = ds.item_ids[item_idx]
            recs.append(
                Recommendation(
                    item_id=item_id,
                    title=ds.title_for(item_id),
                    score=round(float(score), 4),
                )
            )

        return RecommendResponse(
            user_id=user_id,
            model=chosen,
            fallback=fallback,
            recommendations=recs,
        )


engine = Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train all models before the service starts accepting requests."""
    engine.train()
    yield


app = FastAPI(
    title="From-Scratch Recommendation Engine",
    description=(
        "Matrix factorization (SGD), popularity, and item-item kNN "
        "recommenders implemented from scratch in NumPy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check + dataset / model summary."""
    ds = engine.dataset
    if ds is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return HealthResponse(
        status="ok",
        n_users=ds.n_users,
        n_items=ds.n_items,
        n_ratings=ds.n_ratings,
        models=["mf", "popularity", "itemknn"],
    )


@app.get("/movies", response_model=List[Movie])
def movies() -> List[Movie]:
    """Return the full movie catalogue."""
    ds = engine.dataset
    if ds is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return [
        Movie(
            item_id=iid,
            title=ds.title_for(iid),
            genre=ds.genres.get(iid, ""),
        )
        for iid in ds.item_ids
    ]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    """Return ranked recommendations for a user from the chosen model.

    Unknown users transparently fall back to the popularity model (the
    response sets ``fallback=true``).
    """
    if engine.dataset is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return engine.recommend(req.user_id, req.k, req.model)


# --------------------------------------------------------------------------- #
# Tiny HTML demo
# --------------------------------------------------------------------------- #
_DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RecSys Engine demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 1rem; color: #1c1c1e; }
  h1 { font-size: 1.4rem; }
  form { display: flex; gap: .6rem; flex-wrap: wrap; align-items: end;
         margin-bottom: 1.2rem; }
  label { display: flex; flex-direction: column; font-size: .8rem; gap: .2rem; }
  input, select, button { padding: .5rem .6rem; font-size: .95rem; border-radius: 8px;
         border: 1px solid #c7c7cc; }
  button { background: #0a84ff; color: #fff; border: none; cursor: pointer; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: .45rem .5rem; border-bottom: 1px solid #e5e5ea; }
  .note { color: #8e8e93; font-size: .85rem; }
  .pill { background: #ffefc7; padding: .1rem .4rem; border-radius: 6px; }
</style>
</head>
<body>
  <h1>From-scratch recommendation engine</h1>
  <p class="note">MF (SGD) &middot; popularity &middot; item-item kNN, all in NumPy.</p>
  <form id="f">
    <label>User id<input id="user" type="number" value="1" min="1"/></label>
    <label>K<input id="k" type="number" value="10" min="1" max="50"/></label>
    <label>Model
      <select id="model">
        <option value="mf">mf</option>
        <option value="popularity">popularity</option>
        <option value="itemknn">itemknn</option>
      </select>
    </label>
    <button type="submit">Recommend</button>
  </form>
  <p id="status" class="note"></p>
  <table id="out"><thead><tr><th>#</th><th>Title</th><th>Item</th><th>Score</th></tr></thead>
  <tbody></tbody></table>
<script>
const f = document.getElementById('f');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = {
    user_id: parseInt(document.getElementById('user').value, 10),
    k: parseInt(document.getElementById('k').value, 10),
    model: document.getElementById('model').value,
  };
  const res = await fetch('/recommend', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const data = await res.json();
  const tb = document.querySelector('#out tbody');
  tb.innerHTML = '';
  const status = document.getElementById('status');
  status.innerHTML = `model: <b>${data.model}</b>` +
    (data.fallback ? ' <span class="pill">fell back to popularity</span>' : '');
  (data.recommendations || []).forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${idx+1}</td><td>${r.title}</td><td>${r.item_id}</td>` +
                   `<td>${r.score.toFixed(3)}</td>`;
    tb.appendChild(tr);
  });
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
    """Serve the tiny interactive HTML demo."""
    return _DEMO_HTML


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
