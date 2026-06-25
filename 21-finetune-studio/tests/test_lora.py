"""Tests that PROVE the LoRA core math is correct.

Each test maps to a requirement in the spec:
  (1) shapes        — adapter matrices and forward output have the right shapes
  (2) identity init — with B=0 the LoRA output equals the base Linear output
  (3) known math    — with set A/B, forward == base + scaling*(x@A^T@B^T)
  (4) merge         — merged plain Linear reproduces the LoRA forward (allclose)
  extra             — injection wraps the right layers and freezes the base
"""

import torch
import torch.nn as nn

from core.lora import (
    LoRALinear,
    inject_lora,
    iter_lora_layers,
    merge_all,
    trainable_adapter_parameters,
)


def _base_linear(in_f=6, out_f=4, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(in_f, out_f, bias=True)
    # Randomize so base output is non-trivial.
    nn.init.normal_(lin.weight, std=0.5)
    nn.init.normal_(lin.bias, std=0.5)
    return lin


# (1) shapes -----------------------------------------------------------------
def test_lora_shapes():
    base = _base_linear(6, 4)
    lora = LoRALinear(base, r=3, alpha=6.0)
    assert lora.lora_A.shape == (3, 6)        # [r, in]
    assert lora.lora_B.shape == (4, 3)        # [out, r]
    x = torch.randn(2, 5, 6)                  # [B, T, in]
    y = lora(x)
    assert y.shape == (2, 5, 4)               # [B, T, out]
    assert lora.num_adapter_params() == 3 * 6 + 4 * 3


# (2) identity at init (B = 0) ----------------------------------------------
def test_identity_at_init():
    base = _base_linear(6, 4, seed=1)
    lora = LoRALinear(base, r=4, alpha=8.0)   # reset_adapter sets B = 0
    x = torch.randn(3, 7, 6)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


# (3) forward matches independently-computed base + scaling*(x A^T B^T) ------
def test_forward_known_values():
    base = _base_linear(5, 3, seed=2)
    r, alpha = 2, 10.0
    lora = LoRALinear(base, r=r, alpha=alpha, dropout=0.0)
    lora.eval()  # ensure dropout (Identity here anyway) is inert

    torch.manual_seed(99)
    with torch.no_grad():
        lora.lora_A.copy_(torch.randn(r, 5))
        lora.lora_B.copy_(torch.randn(3, r))

    x = torch.randn(4, 5)
    scaling = alpha / r
    expected = base(x) + scaling * (x @ lora_A_T(lora) @ lora_B_T(lora))
    assert torch.allclose(lora(x), expected, atol=1e-5)


def lora_A_T(lora):
    return lora.lora_A.t()   # [in, r]


def lora_B_T(lora):
    return lora.lora_B.t()   # [r, out]


# (4) merge folds adapter; merged forward == unmerged forward ----------------
def test_merge_equivalence():
    base = _base_linear(7, 5, seed=3)
    lora = LoRALinear(base, r=3, alpha=9.0)
    lora.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        lora.lora_A.copy_(torch.randn(3, 7))
        lora.lora_B.copy_(torch.randn(5, 3))

    x = torch.randn(6, 7)
    before = lora(x).clone()
    lora.merge()
    assert lora.merged is True
    after = lora(x)
    assert torch.allclose(before, after, atol=1e-5)

    # Unmerge restores trainable behavior.
    lora.unmerge()
    assert lora.merged is False
    assert torch.allclose(lora(x), before, atol=1e-5)


def test_merge_is_idempotent():
    base = _base_linear(4, 4, seed=5)
    lora = LoRALinear(base, r=2, alpha=4.0)
    with torch.no_grad():
        lora.lora_B.copy_(torch.randn(4, 2))
    w0 = base.weight.detach().clone()
    lora.merge()
    w1 = base.weight.detach().clone()
    lora.merge()  # second merge must be a no-op
    w2 = base.weight.detach().clone()
    assert torch.allclose(w1, w2)
    assert not torch.allclose(w0, w1)  # first merge did change it


# extra: injection wraps target layers and freezes base ----------------------
def test_injection_and_freeze():
    model = nn.Sequential()
    model.add_module("q_proj", nn.Linear(8, 8, bias=False))
    model.add_module("k_proj", nn.Linear(8, 8, bias=False))
    model.add_module("v_proj", nn.Linear(8, 8, bias=False))

    n = inject_lora(model, target_modules=["q_proj", "v_proj"], r=2, alpha=4.0)
    assert n == 2
    names = [name for name, _ in iter_lora_layers(model)]
    assert sorted(names) == ["q_proj", "v_proj"]

    # Base weights frozen, adapters trainable.
    for _, layer in iter_lora_layers(model):
        assert layer.base.weight.requires_grad is False
        assert layer.lora_A.requires_grad is True
        assert layer.lora_B.requires_grad is True

    assert sum(1 for _ in trainable_adapter_parameters(model)) == 4  # 2 layers * (A,B)
    assert merge_all(model) == 2
