"""Tests for data formatting/masking, config validation, and the trainer loop."""

import pytest
import torch

from core.config import TrainingConfig
from core.data import (
    IGNORE_INDEX,
    AlpacaFormatter,
    CharTokenizer,
    build_training_batch,
    encode_with_mask,
    format_example,
)
from core.model import TinyCausalLM
from core.trainer import LoRATrainer, causal_lm_loss


RECORDS = [
    {"instruction": "Say hi", "output": "hello"},
    {"instruction": "Add", "input": "2+2", "output": "4"},
    {"instruction": "Repeat", "output": "abc abc"},
]


# --- config validation ------------------------------------------------------
def test_config_validation_and_scaling():
    cfg = TrainingConfig(r=8, alpha=16.0)
    assert cfg.scaling == 2.0
    assert cfg.lr_at(0) == cfg.lr  # no warmup by default

    warm = TrainingConfig(lr=1.0, warmup_steps=4)
    assert warm.lr_at(0) == pytest.approx(0.25)
    assert warm.lr_at(3) == pytest.approx(1.0)
    assert warm.lr_at(100) == pytest.approx(1.0)

    for bad in [dict(lr=0), dict(r=0), dict(alpha=0), dict(dropout=1.0), dict(epochs=0)]:
        with pytest.raises(ValueError):
            TrainingConfig(**bad)


def test_config_from_dict_ignores_unknown():
    cfg = TrainingConfig.from_dict({"lr": 0.01, "totally_unknown": 5})
    assert cfg.lr == 0.01


# --- formatting + label masking --------------------------------------------
def test_alpaca_template_with_and_without_input():
    fmt = AlpacaFormatter()
    p1 = fmt.prompt("Do thing")
    assert "### Input:" not in p1 and "### Response:" in p1
    p2 = fmt.prompt("Do thing", "context")
    assert "### Input:" in p2 and "context" in p2


def test_label_masking_only_supervises_response():
    tok = CharTokenizer.from_corpus(["".join(chr(c) for c in range(32, 127))])
    ex = format_example({"instruction": "Hi", "output": "yo"}, AlpacaFormatter())
    input_ids, labels = encode_with_mask(ex, tok, max_seq_len=512)

    prompt_len = len(tok.encode(ex.prompt_text))
    # All prompt positions masked.
    assert all(l == IGNORE_INDEX for l in labels[:prompt_len])
    # Response positions are supervised (not all masked).
    assert any(l != IGNORE_INDEX for l in labels[prompt_len:])
    # Final supervised token is EOS.
    assert labels[-1] == tok.eos_id


def test_build_batch_padding_shapes():
    tok = CharTokenizer.from_corpus(["".join(chr(c) for c in range(32, 127))])
    batch = build_training_batch(RECORDS, tok, max_seq_len=256)
    B, T = batch["input_ids"].shape
    assert B == 3
    assert batch["labels"].shape == (B, T)
    assert batch["attention_mask"].shape == (B, T)
    # Pad positions in labels are ignored.
    pad_positions = batch["attention_mask"] == 0
    assert (batch["labels"][pad_positions] == IGNORE_INDEX).all()


# --- trainer: freezes base, trains only adapters, loss decreases ------------
def _make_trainer(epochs=8):
    tok = CharTokenizer.from_corpus(
        ["".join(chr(c) for c in range(32, 127))] + [r["output"] for r in RECORDS]
    )
    cfg = TrainingConfig(
        lr=5e-2, r=4, alpha=8.0, epochs=epochs, batch_size=2,
        max_seq_len=128, target_modules=["q_proj", "v_proj"], seed=0,
    )
    model = TinyCausalLM(vocab_size=tok.vocab_size, d_model=32, n_layers=2, seed=0)
    return LoRATrainer(model, cfg, tok), model


def test_trainer_freezes_base_only_trains_adapters():
    trainer, model = _make_trainer(epochs=1)
    stats = trainer.param_stats()
    assert stats["trainable_params"] > 0
    assert stats["frozen_params"] > stats["trainable_params"]
    assert stats["wrapped_layers"] == 4  # q_proj + v_proj across 2 layers

    # Snapshot a base weight; ensure it doesn't move after a step.
    base_w = None
    for _, layer in trainer.model.named_modules():
        if hasattr(layer, "base") and hasattr(layer.base, "weight"):
            base_w = layer.base.weight.detach().clone()
            ref_layer = layer
            break
    assert base_w is not None

    batch = build_training_batch(RECORDS, trainer.tokenizer, 128)
    trainer.step(batch)
    assert torch.allclose(ref_layer.base.weight, base_w)  # base frozen


def test_trainer_loss_decreases():
    trainer, _ = _make_trainer(epochs=12)
    summary = trainer.train(RECORDS)
    assert summary["steps"] > 0
    assert summary["initial_loss"] is not None
    assert summary["final_loss"] < summary["initial_loss"]  # learning happened


def test_causal_lm_loss_shift_and_mask():
    # All labels masked -> loss is NaN (no supervised tokens), proving masking.
    logits = torch.randn(1, 4, 5)
    labels = torch.full((1, 4), IGNORE_INDEX)
    loss = causal_lm_loss(logits, labels)
    assert torch.isnan(loss)
