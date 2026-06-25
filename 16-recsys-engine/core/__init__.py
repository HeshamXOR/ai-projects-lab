"""Recommendation engine core.

A from-scratch recommender system implemented in NumPy. The public surface
exposes the three models, the offline evaluation harness, and the dataset
loader.

Modules
-------
- :mod:`core.data`        -- dataset generation + loading
- :mod:`core.mf`          -- matrix factorization (SGD, optional ALS)
- :mod:`core.popularity`  -- shrinkage popularity baseline
- :mod:`core.itemknn`     -- item-item cosine kNN
- :mod:`core.eval`        -- Precision@K / Recall@K / NDCG@K harness
"""

from core.data import Dataset, load_dataset
from core.eval import (
    EvalResult,
    evaluate_model,
    leave_one_out_split,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from core.itemknn import ItemKNN
from core.mf import MatrixFactorization
from core.popularity import PopularityRecommender

__all__ = [
    "Dataset",
    "load_dataset",
    "MatrixFactorization",
    "PopularityRecommender",
    "ItemKNN",
    "EvalResult",
    "evaluate_model",
    "leave_one_out_split",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]

__version__ = "1.0.0"
