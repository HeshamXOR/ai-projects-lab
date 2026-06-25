"""LoRA (Low-Rank Adaptation) — the adapter math, implemented from scratch.

The WHY of LoRA: a pretrained linear layer holds a weight W in R^{out x in}.
Full fine-tuning learns a dense update Delta-W of the same shape — millions of
parameters per layer. LoRA's insight is that the *update* needed to adapt a model
to a new task has low intrinsic rank, so we factor it:

        Delta-W  =  (alpha / r) * B @ A

    where  A in R^{r x in},  B in R^{out x r},  and  r << min(in, out).

Only A and B are trained; the base W stays frozen. With r=8 on a 4096x4096 layer
that is 8*4096*2 = 65,536 trainable params instead of 16.7M — a ~256x reduction.

The forward pass of the wrapped layer is therefore:

        y  =  x W^T  +  (alpha / r) * (dropout(x) A^T) B^T   (+ bias)

Two design details that matter and are implemented here explicitly:

  1. ZERO-INIT OF B. At t=0 we want the adapted model to behave *exactly* like
     the base model, so that training starts from the pretrained solution and the
     scaling alpha/r doesn't shock the outputs. We achieve this by initializing
     B = 0 (and A ~ small random). Then B @ A = 0, so y == base output at init.
     Test (2) proves this.

  2. MERGE. For inference we can fold the adapter back into the base weight:
     W_merged = W + (alpha/r) * B @ A. After merging, a single plain Linear
     reproduces the LoRA forward with zero extra latency. Test (4) proves the
     merged linear matches the un-merged LoRA forward to floating-point tolerance.

Everything below is hand-written nn.Module math. We do NOT import peft / any LoRA
library — the factorization, scaling, dropout placement, init, and merge are all
spelled out so the mechanism is fully visible.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """An nn.Linear wrapped with a trainable low-rank adapter.

    Wraps (not subclasses) a base Linear so we can inject this around weights
    that already exist in a pretrained model. The base weight is frozen; only the
    adapter matrices A and B carry gradients.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        *,
        freeze_base: bool = True,
    ) -> None:
        """Wrap an existing nn.Linear with a rank-r adapter.

        Args:
            base: The pretrained linear layer to adapt. Its weight is kept.
            r: Adapter rank (inner dimension). Must be >= 1.
            alpha: Scaling numerator; effective scale is alpha / r.
            dropout: Dropout prob applied to the adapter *input* (regularization
                that only affects the adapter path, never the frozen base path).
            freeze_base: If True (default), base.weight / base.bias get
                requires_grad=False so the optimizer never touches them.
        """
        super().__init__()
        if r < 1:
            raise ValueError(f"LoRA rank r must be >= 1, got {r}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {alpha}")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.merged = False

        # Freeze the base path so .parameters() that require grad == adapter only.
        if freeze_base:
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

        # Adapter factors. Shapes chosen so (B @ A) matches base.weight [out, in].
        #   A: [r, in]   B: [out, r]   =>  B @ A: [out, in]
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r))

        # Dropout on the adapter input only. nn.Dropout is identity in eval mode.
        self.lora_dropout: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )

        self.reset_adapter()

    # ---------------------------------------------------------------- #
    # Initialization                                                   #
    # ---------------------------------------------------------------- #
    def reset_adapter(self) -> None:
        """(Re)initialize adapter: A ~ Kaiming-uniform, B = 0.

        Kaiming-uniform on A keeps the pre-multiply activations well-scaled; B=0
        guarantees the adapter contributes nothing at init (identity start). This
        is the standard LoRA init and is exactly what makes test (2) hold.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    # ---------------------------------------------------------------- #
    # Forward                                                          #
    # ---------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = base(x) + scaling * (dropout(x) @ A^T) @ B^T.

        If the adapter has been merged into the base weight, the adapter path is
        skipped (it's already inside base) to avoid double-counting.
        """
        base_out = self.base(x)
        if self.merged:
            return base_out

        # adapter path: x[..., in] @ A^T[in, r] -> [..., r] @ B^T[r, out]
        dropped = self.lora_dropout(x)
        after_a = F.linear(dropped, self.lora_A)        # [..., r]
        after_b = F.linear(after_a, self.lora_B)        # [..., out]
        return base_out + self.scaling * after_b

    # ---------------------------------------------------------------- #
    # Adapter weight materialization + merge                           #
    # ---------------------------------------------------------------- #
    def delta_weight(self) -> torch.Tensor:
        """The dense update this adapter represents: scaling * (B @ A), [out, in]."""
        return self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight: W <- W + scaling*(B@A).

        After this the layer is a plain Linear at inference time (no adapter
        FLOPs). Idempotent-guarded so a double merge can't corrupt the weight.
        """
        if self.merged:
            return
        self.base.weight.data += self.delta_weight().to(self.base.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo merge(): W <- W - scaling*(B@A). Restores the trainable form."""
        if not self.merged:
            return
        self.base.weight.data -= self.delta_weight().to(self.base.weight.dtype)
        self.merged = False

    # ---------------------------------------------------------------- #
    # Introspection helpers                                            #
    # ---------------------------------------------------------------- #
    def adapter_parameters(self):
        """Yield only the trainable adapter params (A and B)."""
        yield self.lora_A
        yield self.lora_B

    def num_adapter_params(self) -> int:
        """Count of trainable adapter scalars: r*(in + out)."""
        return self.lora_A.numel() + self.lora_B.numel()

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"
        )


# ============================================================================
# Injection: walk a model and replace target nn.Linear modules with LoRALinear.
# ============================================================================
def inject_lora(
    model: nn.Module,
    target_modules,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Replace matching nn.Linear submodules in `model` with LoRALinear in place.

    A submodule is targeted if any string in `target_modules` is a substring of
    its dotted attribute name (e.g. "q_proj" matches "layers.0.attn.q_proj").

    Args:
        model: The (frozen) base model to adapt.
        target_modules: Iterable of name-substrings to match.
        r, alpha, dropout: Adapter hyperparameters (see LoRALinear).

    Returns:
        The number of layers wrapped.
    """
    targets = list(target_modules)
    wrapped = 0

    # Collect first to avoid mutating during traversal.
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            to_replace.append((name, module))

    for name, module in to_replace:
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        wrapped += 1

    return wrapped


def _resolve_parent(model: nn.Module, dotted: str):
    """Return (parent_module, final_attr_name) for a dotted submodule path."""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def iter_lora_layers(model: nn.Module):
    """Yield (name, LoRALinear) pairs for every adapter in the model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def merge_all(model: nn.Module) -> int:
    """Merge every adapter in the model. Returns count merged."""
    n = 0
    for _, layer in iter_lora_layers(model):
        layer.merge()
        n += 1
    return n


def trainable_adapter_parameters(model: nn.Module):
    """Yield all adapter parameters across the model (for the optimizer)."""
    for _, layer in iter_lora_layers(model):
        yield from layer.adapter_parameters()
