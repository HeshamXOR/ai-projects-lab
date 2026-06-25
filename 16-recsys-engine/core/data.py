"""Dataset generation and loading for the recommendation engine.

The bundled dataset is a small MovieLens-style ratings matrix. Crucially it is
*generated from a genuine latent-factor model* (known user/item factors plus
biases and a small amount of noise). That means there is real low-rank
structure for matrix factorization to discover, so an MF model can plausibly
beat a popularity baseline -- which is exactly what the tests assert.

The generator is deterministic given a seed, so the bundled CSV files are
reproducible. Files live in ``data/``:

- ``movies.csv``   -- item_id, title, genre
- ``ratings.csv``  -- user_id, item_id, rating

Run ``python -m core.data`` to (re)generate the bundled files.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "data"))
MOVIES_CSV = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

RATING_MIN = 1.0
RATING_MAX = 5.0

_GENRES = [
    "Action",
    "Comedy",
    "Drama",
    "Sci-Fi",
    "Romance",
    "Thriller",
    "Animation",
    "Horror",
]

# A pool of plausible-looking movie titles. We index into this so the bundled
# data reads like a real catalogue rather than "Movie 1, Movie 2, ...".
_TITLE_POOL = [
    "The Last Horizon", "Midnight Circuit", "Paper Lanterns", "Echoes of Tomorrow",
    "Crimson Harbor", "The Quiet Algorithm", "Velvet Thunder", "Northbound",
    "A Study in Static", "Glass Mountain", "The Ninth Gate House", "Solar Drift",
    "Whispering Pines", "Concrete Gardens", "The Pixel Heist", "Lunar Tide",
    "Brass and Bone", "The Forgotten Reel", "Saltwater Kings", "Neon Cathedral",
    "The Map of Small Things", "Ironwood", "Parallel Hearts", "The Long Commute",
    "Ashes of Verona", "Quantum Lullaby", "The Paper Crown", "Driftwood County",
    "Silent Frequency", "The Glasshouse Rules", "Cobalt Dreams", "The Final Draft",
    "Marrowbone Lane", "The Cartographer", "Hollow Sun", "Tin Can Symphony",
    "The Velveteen Spy", "Breakwater", "The Inheritance Engine", "Ghostlight",
    "Sunday at the Terminus", "The Origami War", "Riverbend", "The Clockmaker's Son",
    "Static Bloom", "The Long Winter Count", "Wireframe", "The Salt Road",
    "Amber Protocol", "The Understudy",
]


@dataclass
class Dataset:
    """In-memory representation of the ratings dataset.

    Ratings are stored both as a contiguous triplet array (for SGD-style
    iteration) and as a dense user-item matrix (for kNN / popularity). All
    indexing uses *internal* contiguous indices; the original (possibly
    sparse) ids are recoverable via the mapping dictionaries.

    Attributes
    ----------
    user_ids, item_ids:
        Sorted lists of the original external ids.
    user_index, item_index:
        Maps from external id -> internal contiguous index [0, n).
    titles:
        Map external item id -> title string.
    genres:
        Map external item id -> genre string.
    ratings:
        ``(n_obs, 3)`` float array of (user_idx, item_idx, rating).
    """

    user_ids: List[int]
    item_ids: List[int]
    user_index: Dict[int, int]
    item_index: Dict[int, int]
    titles: Dict[int, str]
    genres: Dict[int, str] = field(default_factory=dict)
    ratings: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    # ------------------------------------------------------------------ #
    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_ratings(self) -> int:
        return int(self.ratings.shape[0])

    def title_for(self, item_id: int) -> str:
        """Return the title for an external item id (or a fallback)."""
        return self.titles.get(item_id, f"Item {item_id}")

    def user_item_matrix(self, fill: float = 0.0) -> np.ndarray:
        """Build a dense ``(n_users, n_items)`` matrix.

        Unobserved entries are set to ``fill`` (0 by default, which the
        item-kNN and popularity models treat as "missing").
        """
        mat = np.full((self.n_users, self.n_items), fill, dtype=np.float64)
        u = self.ratings[:, 0].astype(int)
        i = self.ratings[:, 1].astype(int)
        mat[u, i] = self.ratings[:, 2]
        return mat

    def known_items_for_user(self, user_idx: int) -> np.ndarray:
        """Internal item indices the given user has already rated."""
        mask = self.ratings[:, 0].astype(int) == user_idx
        return np.unique(self.ratings[mask, 1].astype(int))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_synthetic(
    n_users: int = 50,
    n_items: int = 40,
    n_factors: int = 4,
    density: float = 0.22,
    noise: float = 0.35,
    seed: int = 7,
) -> Tuple[List[Tuple[int, int, str, str]], List[Tuple[int, int, float]]]:
    """Generate movies + ratings from a true latent-factor model.

    The data-generating process mirrors the MF model the engine learns::

        r_ui = clip( mu + b_u + b_i + p_u . q_i + eps )

    with ``p_u, q_i`` drawn from a low-rank Gaussian, biases drawn per
    user/item, and ``eps`` small Gaussian noise. Because the signal is
    genuinely low-rank, a rank-``n_factors`` MF can recover it and should beat
    a popularity baseline on held-out ranking quality.

    Returns
    -------
    movies:
        list of ``(item_id, _unused, title, genre)`` -- second field is a
        placeholder kept for tuple symmetry and ignored by the writer.
    ratings:
        list of ``(user_id, item_id, rating)``.
    """
    rng = np.random.default_rng(seed)

    # True latent factors. Scale chosen so dot products are O(1).
    P = rng.normal(0.0, 0.55, size=(n_users, n_factors))
    Q = rng.normal(0.0, 0.55, size=(n_items, n_factors))
    b_u = rng.normal(0.0, 0.45, size=n_users)
    b_i = rng.normal(0.0, 0.65, size=n_items)
    mu = 3.4

    movies: List[Tuple[int, int, str, str]] = []
    for j in range(n_items):
        title = _TITLE_POOL[j % len(_TITLE_POOL)]
        if j >= len(_TITLE_POOL):
            title = f"{title} ({j // len(_TITLE_POOL) + 1})"
        genre = _GENRES[j % len(_GENRES)]
        movies.append((j + 1, 0, title, genre))

    ratings: List[Tuple[int, int, float]] = []
    for u in range(n_users):
        # Each user rates a random subset of items; guarantee a minimum so no
        # user is empty (important for leave-one-out evaluation).
        n_rate = max(6, int(rng.binomial(n_items, density)))
        n_rate = min(n_rate, n_items)
        items = rng.choice(n_items, size=n_rate, replace=False)
        for it in items:
            true = mu + b_u[u] + b_i[it] + float(P[u] @ Q[it])
            r = true + rng.normal(0.0, noise)
            r = float(np.clip(round(r * 2) / 2.0, RATING_MIN, RATING_MAX))
            ratings.append((u + 1, int(it) + 1, r))

    return movies, ratings


def write_dataset(
    movies: List[Tuple[int, int, str, str]],
    ratings: List[Tuple[int, int, float]],
    data_dir: str = DATA_DIR,
) -> None:
    """Persist generated movies/ratings to CSV in ``data_dir``."""
    os.makedirs(data_dir, exist_ok=True)
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    with open(movies_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "title", "genre"])
        for item_id, _unused, title, genre in movies:
            w.writerow([item_id, title, genre])

    with open(ratings_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "item_id", "rating"])
        for user_id, item_id, rating in ratings:
            w.writerow([user_id, item_id, rating])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(
    movies_csv: str = MOVIES_CSV,
    ratings_csv: str = RATINGS_CSV,
) -> Dataset:
    """Load the bundled (or any compatible) CSV dataset into a :class:`Dataset`.

    Builds contiguous internal indices for users and items, the title/genre
    maps, and the ``(n_obs, 3)`` ratings triplet array.

    Raises
    ------
    FileNotFoundError
        If either CSV is missing.
    """
    if not os.path.exists(ratings_csv):
        raise FileNotFoundError(
            f"ratings file not found: {ratings_csv}. "
            f"Run `python -m core.data` to generate the bundled dataset."
        )

    titles: Dict[int, str] = {}
    genres: Dict[int, str] = {}
    if os.path.exists(movies_csv):
        with open(movies_csv, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                iid = int(row["item_id"])
                titles[iid] = row.get("title", f"Item {iid}")
                genres[iid] = row.get("genre", "")

    raw: List[Tuple[int, int, float]] = []
    with open(ratings_csv, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw.append(
                (int(row["user_id"]), int(row["item_id"]), float(row["rating"]))
            )

    if not raw:
        raise ValueError(f"ratings file is empty: {ratings_csv}")

    user_ids = sorted({r[0] for r in raw})
    # Items come from ratings *and* the movie catalogue, so the catalogue may
    # advertise items nobody has rated yet (still recommendable by popularity
    # fallback / cold handling).
    item_ids = sorted({r[1] for r in raw} | set(titles.keys()))

    user_index = {uid: idx for idx, uid in enumerate(user_ids)}
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}

    triplets = np.empty((len(raw), 3), dtype=np.float64)
    for k, (uid, iid, rating) in enumerate(raw):
        triplets[k, 0] = user_index[uid]
        triplets[k, 1] = item_index[iid]
        triplets[k, 2] = rating

    # Fill any missing titles for items that only appear in ratings.
    for iid in item_ids:
        titles.setdefault(iid, f"Item {iid}")
        genres.setdefault(iid, "")

    return Dataset(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
        titles=titles,
        genres=genres,
        ratings=triplets,
    )


def ensure_dataset(seed: int = 7) -> Dataset:
    """Load the bundled dataset, generating it first if absent."""
    if not (os.path.exists(MOVIES_CSV) and os.path.exists(RATINGS_CSV)):
        movies, ratings = generate_synthetic(seed=seed)
        write_dataset(movies, ratings)
    return load_dataset()


def _main() -> None:
    movies, ratings = generate_synthetic()
    write_dataset(movies, ratings)
    print(
        f"Wrote {len(movies)} movies and {len(ratings)} ratings to {DATA_DIR}"
    )


if __name__ == "__main__":
    _main()
