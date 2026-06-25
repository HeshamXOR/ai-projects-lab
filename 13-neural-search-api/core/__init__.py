"""Neural Search API core package.

A hybrid vector-search microservice whose hard core is implemented from
scratch in pure Python / NumPy:

* :mod:`core.ann`       -- an HNSW-style approximate nearest-neighbor index.
* :mod:`core.bm25`      -- a BM25 lexical index (tokenizer, TF, IDF, scoring).
* :mod:`core.fusion`    -- reciprocal-rank fusion of result lists.
* :mod:`core.embed`     -- a deterministic from-scratch text embedder.
* :mod:`core.store`     -- on-disk persistence for the whole index.
* :mod:`core.retriever` -- a hybrid retriever tying it all together.

Pretrained models (sentence-transformers) are wired in as one *optional*
component in :mod:`core.embed`; everything else runs with zero external
model dependencies.
"""

from __future__ import annotations

__all__ = [
    "ann",
    "bm25",
    "fusion",
    "embed",
    "store",
    "retriever",
]

__version__ = "1.0.0"
