"""Dataset formatting + a dependency-free tokenizer for instruction tuning.

The WHY: instruction fine-tuning takes records like
    {"instruction": "...", "input": "...", "output": "..."}
and must turn them into (1) a single flat prompt string in a consistent template
and (2) integer token ids with a LABEL MASK so the loss is computed *only* over
the response tokens. If we trained on the prompt tokens too, the model would
waste capacity learning to reproduce the instruction it was just given instead
of learning the behavior we actually want (the response).

This module provides:
  * AlpacaFormatter      — the classic Alpaca prompt template, with/without input.
  * CharTokenizer        — a tiny char-level tokenizer so tests and the demo run
                           with zero external downloads. Implements the same
                           encode/decode/pad interface a real HF tokenizer exposes,
                           so a production deployment can swap in a BPE tokenizer
                           without touching the rest of the pipeline.
  * format_example       — record -> (prompt_text, full_text)
  * build_training_batch — list of records -> padded input_ids + labels (with the
                           prompt region masked out via IGNORE_INDEX).

IGNORE_INDEX (-100) is the conventional "ignore this position" label that a
cross-entropy loss skips; our eval/trainer respect it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import torch


# Standard "skip this token in the loss" sentinel (matches torch's default).
IGNORE_INDEX: int = -100


# ============================================================================
# Tokenizer interface — anything with this shape is a drop-in (HF-compatible).
# ============================================================================
class Tokenizer(Protocol):
    """Minimal tokenizer protocol the pipeline depends on."""

    pad_id: int
    eos_id: int

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, add_eos: bool = False) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class CharTokenizer:
    """A deterministic character-level tokenizer (no external deps).

    Vocab layout: id 0 = PAD, id 1 = EOS, id 2 = UNK, then one id per known
    character. Building the vocab from a corpus keeps the table small; unknown
    chars at encode time fall back to UNK so it never crashes on new input.
    """

    PAD = "<pad>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, charset: Optional[Sequence[str]] = None) -> None:
        specials = [self.PAD, self.EOS, self.UNK]
        chars = list(charset) if charset is not None else []
        # Deduplicate while preserving order for determinism.
        seen = set()
        ordered_chars = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                ordered_chars.append(c)

        self._itos: List[str] = specials + ordered_chars
        self._stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self._itos)}
        self.pad_id = self._stoi[self.PAD]
        self.eos_id = self._stoi[self.EOS]
        self.unk_id = self._stoi[self.UNK]

    @classmethod
    def from_corpus(cls, texts: Sequence[str]) -> "CharTokenizer":
        """Build a tokenizer whose vocab covers every char seen in `texts`."""
        charset: List[str] = []
        seen = set()
        for t in texts:
            for ch in t:
                if ch not in seen:
                    seen.add(ch)
                    charset.append(ch)
        return cls(sorted(charset))

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        ids = [self._stoi.get(ch, self.unk_id) for ch in text]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out: List[str] = []
        for i in ids:
            if i == self.pad_id or i == self.eos_id:
                continue
            tok = self._itos[i] if 0 <= i < len(self._itos) else self.UNK
            if tok in (self.PAD, self.EOS, self.UNK):
                continue
            out.append(tok)
        return "".join(out)


# ============================================================================
# Prompt template (Alpaca-style)
# ============================================================================
@dataclass
class AlpacaFormatter:
    """Render instruction records into the Alpaca prompt template.

    The template differs depending on whether an `input` field is present, which
    is exactly how the original Alpaca dataset is structured.
    """

    with_input_header: str = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
    no_input_header: str = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    )

    def prompt(self, instruction: str, input_text: str = "") -> str:
        """Return the prompt portion (everything up to and including Response:)."""
        if input_text and input_text.strip():
            return self.with_input_header.format(
                instruction=instruction.strip(), input=input_text.strip()
            )
        return self.no_input_header.format(instruction=instruction.strip())


@dataclass
class FormattedExample:
    """A single example split into its prompt and response text."""

    prompt_text: str
    response_text: str

    @property
    def full_text(self) -> str:
        return self.prompt_text + self.response_text


def format_example(
    record: Dict[str, str],
    formatter: Optional[AlpacaFormatter] = None,
) -> FormattedExample:
    """Turn one instruction/input/output record into a FormattedExample.

    Raises:
        KeyError / ValueError if required fields are missing or empty.
    """
    formatter = formatter or AlpacaFormatter()
    if "instruction" not in record:
        raise KeyError("record missing required 'instruction' field")
    response = record.get("output", record.get("response", ""))
    if not response:
        raise ValueError("record must have a non-empty 'output'/'response'")
    prompt_text = formatter.prompt(record["instruction"], record.get("input", ""))
    return FormattedExample(prompt_text=prompt_text, response_text=response)


# ============================================================================
# Tokenization + label masking + padding into a batch
# ============================================================================
def encode_with_mask(
    example: FormattedExample,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    """Encode a FormattedExample into (input_ids, labels) with prompt masked.

    The label at position t is the token the model should predict at t (we keep
    them aligned to input_ids here; the trainer does the standard shift-by-one).
    Prompt positions get IGNORE_INDEX so they contribute zero loss; response
    positions keep their true token id. An EOS is appended so the model learns to
    stop, and that EOS *is* supervised (part of the response label region).
    """
    prompt_ids = tokenizer.encode(example.prompt_text, add_eos=False)
    response_ids = tokenizer.encode(example.response_text, add_eos=True)

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

    # Truncate from the left of the *prompt* if needed so we keep the response.
    if len(input_ids) > max_seq_len:
        overflow = len(input_ids) - max_seq_len
        # Drop from the front (prompt side) first.
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return input_ids, labels


def build_training_batch(
    records: Sequence[Dict[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
    formatter: Optional[AlpacaFormatter] = None,
) -> Dict[str, torch.Tensor]:
    """Format, tokenize, mask, and pad a list of records into tensors.

    Returns a dict with:
        input_ids: LongTensor [B, T]   (padded with tokenizer.pad_id)
        labels:    LongTensor [B, T]   (prompt + pad positions = IGNORE_INDEX)
        attention_mask: LongTensor [B, T] (1 for real tokens, 0 for pad)
    """
    formatter = formatter or AlpacaFormatter()
    encoded: List[Tuple[List[int], List[int]]] = []
    for rec in records:
        ex = format_example(rec, formatter)
        encoded.append(encode_with_mask(ex, tokenizer, max_seq_len))

    max_len = max((len(ids) for ids, _ in encoded), default=1)
    max_len = max(max_len, 1)

    batch_input: List[List[int]] = []
    batch_labels: List[List[int]] = []
    batch_mask: List[List[int]] = []
    for input_ids, labels in encoded:
        pad_n = max_len - len(input_ids)
        batch_input.append(input_ids + [tokenizer.pad_id] * pad_n)
        batch_labels.append(labels + [IGNORE_INDEX] * pad_n)
        batch_mask.append([1] * len(input_ids) + [0] * pad_n)

    return {
        "input_ids": torch.tensor(batch_input, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
    }


def iter_batches(
    records: Sequence[Dict[str, str]],
    tokenizer: Tokenizer,
    max_seq_len: int,
    batch_size: int,
    formatter: Optional[AlpacaFormatter] = None,
):
    """Yield padded batches of the given size over the records (no shuffle)."""
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        yield build_training_batch(chunk, tokenizer, max_seq_len, formatter)
