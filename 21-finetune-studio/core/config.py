"""Training configuration — a validated dataclass for a LoRA fine-tune run.

The WHY: fine-tuning has a wide hyperparameter surface (learning rate, adapter
rank, scaling alpha, dropout, epochs, batch size, sequence length, ...). Passing
these around as a loose dict invites silent typos and out-of-range values that
only blow up three hours into a run. Instead we centralize them in one frozen-ish
dataclass that validates itself at construction time, so an invalid config fails
fast at the API boundary rather than mid-training.

We also expose the derived LoRA scaling factor (alpha / r) here because it is a
pure function of the config and several modules need it (the adapter forward
pass, the merge step, and logging), so it belongs with the config rather than
being recomputed inline everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


# ============================================================================
# Defaults that are sane for a tiny CPU smoke-train. Real runs override these.
# ============================================================================
_TARGET_DEFAULT: List[str] = ["q_proj", "v_proj"]


@dataclass
class TrainingConfig:
    """All hyperparameters for one fine-tuning run.

    Attributes:
        lr: Learning rate for the optimizer (Adam-style) applied to adapter params.
        r: LoRA rank — the inner dimension of the low-rank update. Smaller = fewer
           trainable params and stronger regularization.
        alpha: LoRA scaling numerator. The effective update is scaled by alpha / r.
        dropout: Dropout probability applied to the *input* of each adapter.
        epochs: Number of passes over the dataset.
        batch_size: Examples per optimizer step.
        max_seq_len: Truncate/pad formatted examples to this many tokens.
        weight_decay: L2 regularization coefficient on adapter params.
        grad_clip: Max global gradient norm (<= 0 disables clipping).
        seed: RNG seed for reproducible init and shuffling.
        target_modules: Substrings of linear-layer names to wrap with adapters.
        warmup_steps: Steps of linear LR warmup before the constant phase.
    """

    lr: float = 1e-3
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 128
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 0
    target_modules: List[str] = field(default_factory=lambda: list(_TARGET_DEFAULT))
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        self._validate()

    # ------------------------------------------------------------------ #
    # Validation: every numeric field has a physically meaningful range. #
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.r < 1:
            raise ValueError(f"r (rank) must be >= 1, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if not self.target_modules:
            raise ValueError("target_modules must list at least one module name")

    @property
    def scaling(self) -> float:
        """The LoRA scaling factor gamma = alpha / r applied to B @ A."""
        return self.alpha / self.r

    def lr_at(self, step: int) -> float:
        """Learning rate at a given global step (linear warmup, then constant).

        Warmup avoids the large, noisy updates that an untrained adapter would
        otherwise take in the first few steps when gradients are biggest.
        """
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.lr * float(step + 1) / float(self.warmup_steps)
        return self.lr

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON responses / logging."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Build from a (possibly partial) dict, ignoring unknown keys.

        This is what the API layer uses: clients send only the knobs they care
        about and the rest fall back to defaults. Unknown keys are dropped rather
        than raising so that forward-compatible clients don't break old servers.
        """
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
