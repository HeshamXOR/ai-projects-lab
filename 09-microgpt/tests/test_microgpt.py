"""Tests for the from-scratch BPE tokenizer and the GPT model wiring."""

import torch

from bpe import BPE
from model import GPT, GPTConfig


def test_bpe_roundtrips_exactly():
    text = "the quick brown fox jumps over the lazy dog. " * 20
    tok = BPE()
    tok.train(text, vocab_size=400)
    assert tok.decode(tok.encode(text)) == text


def test_bpe_roundtrips_unicode():
    text = "café — naïve — 日本語 — emoji 🚀 " * 10
    tok = BPE()
    tok.train(text, vocab_size=350)
    # round-trip is exact at the byte level
    assert tok.decode(tok.encode(text)) == text


def test_bpe_actually_compresses():
    text = "ababababab " * 100
    tok = BPE()
    tok.train(text, vocab_size=300)
    # repeated "ab" should merge, so encoded length << raw byte length
    assert len(tok.encode(text)) < len(text.encode("utf-8")) // 2


def test_bpe_merges_grow_vocab():
    tok = BPE()
    tok.train("hello world " * 50, vocab_size=280)
    assert tok.vocab_size == 280
    assert len(tok.merges) == 280 - 256


def test_gpt_forward_and_loss_shape():
    cfg = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 128, (4, 16))
    y = torch.randint(0, 128, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 128)
    assert loss.item() > 0
    # one backward step works
    loss.backward()
    assert model.tok_emb.weight.grad is not None


def test_gpt_generate_extends_sequence():
    cfg = GPTConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    ctx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=10)
    assert out.shape == (1, 11)


def test_weight_tying():
    cfg = GPTConfig(vocab_size=64, block_size=8, n_layer=1, n_head=1, n_embd=16)
    model = GPT(cfg)
    # input embedding and output head share the same tensor
    assert model.head.weight is model.tok_emb.weight
