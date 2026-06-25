"""Deterministic, dependency-free text embeddings.

The default embedder is implemented entirely from scratch: it tokenizes
text, hashes tokens into a fixed-width vector (the *hashing trick*),
applies a sub-linear term-frequency weight, and L2-normalizes the result.
Because the hash is deterministic, the same text always maps to the same
vector -- which makes the whole service reproducible and testable without
downloading any model.

An *optional* :class:`SentenceTransformerEmbedder` is provided as a thin
hook around the ``sentence-transformers`` package. It is imported lazily
inside a ``try/except`` so the service runs perfectly without it.

All embedders implement the :class:`Embedder` protocol::

    dim: int
    embed(text: str) -> np.ndarray            # shape (dim,), float32
    embed_batch(texts: Sequence[str]) -> np.ndarray  # shape (n, dim)
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

import numpy as np

# A simple, Unicode-aware word tokenizer shared with the BM25 index.
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase ``text`` and split it into alphanumeric tokens.

    Args:
        text: Arbitrary input string.

    Returns:
        A list of lowercased alphanumeric tokens.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _token_hash(token: str, dim: int) -> int:
    """Map ``token`` to an index in ``[0, dim)`` deterministically.

    Uses a BLAKE2b digest so the result is stable across processes and
    Python invocations (unlike the salted built-in ``hash``).
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _token_sign(token: str) -> float:
    """Return a deterministic +/-1 sign for ``token``.

    Signed hashing reduces the bias introduced by hash collisions: two
    different tokens that collide on the same bucket have a 50% chance of
    partially cancelling rather than always reinforcing each other.
    """
    digest = hashlib.blake2b(
        (token + "#sign").encode("utf-8"), digest_size=1
    ).digest()
    return 1.0 if (digest[0] & 1) else -1.0


class HashingEmbedder:
    """A from-scratch hashing / bag-of-words embedder.

    The embedding for a document is built as follows:

    1. Tokenize the text.
    2. For each unique token, compute a sub-linear TF weight
       ``1 + log(count)``.
    3. Accumulate ``sign(token) * weight`` into bucket ``hash(token)``.
    4. L2-normalize the resulting vector so cosine similarity reduces to a
       plain dot product.

    Args:
        dim: Dimensionality of the output vectors.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.name = f"hashing-{self.dim}"

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a unit-norm vector of shape ``(dim,)``."""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec

        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

        for tok, count in counts.items():
            weight = 1.0 + math.log(count)
            idx = _token_hash(tok, self.dim)
            vec[idx] += _token_sign(tok) * weight

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of strings into an ``(n, dim)`` float32 matrix."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    """Optional hook around ``sentence-transformers``.

    This class is only usable if the ``sentence-transformers`` package is
    installed. Construction raises :class:`RuntimeError` otherwise, so the
    rest of the service can fall back to :class:`HashingEmbedder`.

    Args:
        model_name: Any model id accepted by ``SentenceTransformer``.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:  # Lazy / optional import -- never required to run the service.
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "sentence-transformers is not installed; install it or use "
                "HashingEmbedder instead"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st-{model_name}"

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string with the pretrained model."""
        vec = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(vec, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed many strings with the pretrained model."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(mat, dtype=np.float32)


def get_embedder(prefer_pretrained: bool = False, dim: int = 256):
    """Return an embedder, preferring the optional pretrained one if asked.

    Args:
        prefer_pretrained: If True, try to build a
            :class:`SentenceTransformerEmbedder` and fall back to the
            hashing embedder when unavailable.
        dim: Dimensionality used for the hashing embedder fallback.

    Returns:
        An object implementing the embedder protocol.
    """
    if prefer_pretrained:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            # Fall back silently -- the service must always start.
            return HashingEmbedder(dim=dim)
    return HashingEmbedder(dim=dim)
