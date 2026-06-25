"""From-scratch text tokenizer.

Lowercasing, Unicode-aware normalization, punctuation handling, whitespace
splitting and optional n-gram generation. No external libraries -- only the
standard library (``re``, ``unicodedata``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List


# A token is a maximal run of word characters (letters, digits, underscore)
# OR an apostrophe-joined contraction such as ``don't``. Everything else is a
# separator. We keep the regex deliberately simple and readable.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)

# Characters that look alike are folded so that l33t-speak and homoglyph
# evasion ("st0pid", "1diot") collapse toward their plain forms before the
# classifier ever sees them.
_LEET_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
}


@dataclass
class Tokenizer:
    """Convert raw text into a normalized list of tokens.

    Attributes:
        lowercase: Fold everything to lower case.
        fold_leet: Map common leet-speak substitutions back to letters.
        strip_accents: Remove diacritics via Unicode decomposition.
        ngram_range: Inclusive ``(min_n, max_n)`` range of n-grams to emit.
            ``(1, 1)`` yields plain unigrams; ``(1, 2)`` adds bigrams, etc.
    """

    lowercase: bool = True
    fold_leet: bool = True
    strip_accents: bool = True
    ngram_range: tuple[int, int] = (1, 1)

    def normalize(self, text: str) -> str:
        """Apply character-level normalization to ``text``.

        Order matters: NFKC first so that compatibility characters collapse,
        then accent stripping, then case folding, then leet folding.
        """
        if not isinstance(text, str):
            raise TypeError(f"expected str, got {type(text).__name__}")

        text = unicodedata.normalize("NFKC", text)

        if self.strip_accents:
            decomposed = unicodedata.normalize("NFKD", text)
            text = "".join(c for c in decomposed if not unicodedata.combining(c))

        if self.lowercase:
            text = text.lower()

        if self.fold_leet:
            text = "".join(_LEET_MAP.get(c, c) for c in text)

        return text

    def split(self, text: str) -> List[str]:
        """Return the unigram tokens of ``text`` after normalization."""
        normalized = self.normalize(text)
        return _TOKEN_RE.findall(normalized)

    def _ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Return the list of ``n``-grams (space-joined) from ``tokens``."""
        if n <= 1:
            return list(tokens)
        return [
            " ".join(tokens[i : i + n])
            for i in range(0, len(tokens) - n + 1)
        ]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize ``text`` honoring ``ngram_range``.

        Returns unigrams plus any higher-order n-grams requested. The order is
        deterministic: all 1-grams, then all 2-grams, and so on.
        """
        unigrams = self.split(text)
        min_n, max_n = self.ngram_range
        if min_n < 1:
            raise ValueError("ngram_range minimum must be >= 1")
        if max_n < min_n:
            raise ValueError("ngram_range maximum must be >= minimum")

        out: List[str] = []
        for n in range(min_n, max_n + 1):
            out.extend(self._ngrams(unigrams, n))
        return out

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)
