"""Byte-Pair Encoding tokenizer, trained from scratch.

BPE is how GPT-2/3/4, Llama, and most modern LLMs tokenize text. It starts from
raw bytes and repeatedly merges the most frequent adjacent pair into a new
token, building a vocabulary that balances "short sequences" against "small
vocab." This is the real algorithm, implemented without the HF `tokenizers`
library.

Train:   tok = BPE(); tok.train(text, vocab_size=512)
Encode:  ids = tok.encode("hello world")   # list[int]
Decode:  tok.decode(ids) == "hello world"  # round-trips exactly
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Tuple


class BPE:
    def __init__(self):
        # token id -> bytes. Ids 0..255 are the raw byte values.
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # ordered list of learned merges: (a_id, b_id) -> new_id
        self.merges: Dict[Tuple[int, int], int] = {}

    # ---- training ----
    @staticmethod
    def _get_pair_counts(ids: List[int]) -> Counter:
        counts = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int = 512, verbose: bool = False) -> None:
        """Learn merges until the vocabulary reaches `vocab_size`."""
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for m in range(num_merges):
            counts = self._get_pair_counts(ids)
            if not counts:
                break
            # most frequent adjacent pair
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break  # nothing worth merging
            new_id = 256 + m
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and m % 50 == 0:
                print(f"merge {m}: {pair} -> {new_id} ({counts[pair]}x)")

    # ---- inference ----
    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        # apply merges in the order they were learned (lowest new_id first)
        while len(ids) >= 2:
            counts = self._get_pair_counts(ids)
            # find the learnable pair that was merged earliest
            pair = min(
                (p for p in counts if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if pair is None:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: List[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- persistence ----
    def save(self, path: str) -> None:
        obj = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.merges = {tuple(k): v for k, v in obj["merges"]}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
