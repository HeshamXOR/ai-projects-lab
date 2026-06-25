"""The trainer: inject LoRA, freeze the base, and learn only the adapters.

The WHY: this ties the pieces together into a real training loop. Given a base
CausalLM and a TrainingConfig, we (1) inject LoRA adapters into the configured
linear layers, (2) hand ONLY the adapter parameters to the optimizer so the
frozen base never moves, and (3) run the standard next-token-prediction loop:

    logits = model(input_ids)                 # [B, T, V]
    shift logits/labels by one (predict t+1 from <=t)
    loss   = cross_entropy(shift_logits, shift_labels, ignore_index=-100)

The shift is the crux of causal LM training: position t's prediction is scored
against token t+1, and prompt/pad positions (labels == IGNORE_INDEX) are skipped
so the loss is response-only. We implement the optimizer step explicitly with
gradient clipping and a warmup LR schedule pulled from the config.

step() returns the scalar loss so the API / tests can assert it decreases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainingConfig
from .data import (
    IGNORE_INDEX,
    Tokenizer,
    iter_batches,
)
from .eval import Evaluator, EvalResult
from .lora import inject_lora, iter_lora_layers, trainable_adapter_parameters


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Shifted next-token cross-entropy.

    logits[:, t] predicts labels[:, t+1]; the last logit and first label have no
    partner and are dropped. Positions with label == ignore_index don't count.
    """
    shift_logits = logits[:, :-1, :].contiguous()              # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()                  # [B, T-1]
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


@dataclass
class TrainState:
    """Mutable bookkeeping across the run (for logging / API responses)."""

    step: int = 0
    epoch: int = 0
    losses: List[float] = field(default_factory=list)

    @property
    def last_loss(self) -> Optional[float]:
        return self.losses[-1] if self.losses else None


class LoRATrainer:
    """Owns the model, optimizer, and loop for a single fine-tune run."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer: Tokenizer,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.state = TrainState()

        torch.manual_seed(config.seed)

        # 1) Inject adapters into the targeted linear layers.
        self.num_wrapped = inject_lora(
            model,
            target_modules=config.target_modules,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        if self.num_wrapped == 0:
            raise ValueError(
                "inject_lora wrapped 0 layers — check target_modules "
                f"{config.target_modules} against the model's layer names"
            )

        # 2) Optimizer sees ONLY adapter params. (Base is frozen inside LoRALinear.)
        adapter_params = list(trainable_adapter_parameters(model))
        if not adapter_params:
            raise ValueError("no trainable adapter parameters found after injection")
        self.optimizer = torch.optim.Adam(
            adapter_params, lr=config.lr, weight_decay=config.weight_decay
        )
        self._adapter_params = adapter_params

    # ---------------------------------------------------------------- #
    # Parameter accounting (for the API response)                      #
    # ---------------------------------------------------------------- #
    def param_stats(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self._adapter_params)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "wrapped_layers": self.num_wrapped,
        }

    # ---------------------------------------------------------------- #
    # One optimizer step on a pre-built batch                          #
    # ---------------------------------------------------------------- #
    def step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Run forward/backward/update on one batch; return the scalar loss."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(batch["input_ids"])
        loss = causal_lm_loss(logits, batch["labels"])
        loss.backward()

        if self.config.grad_clip and self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._adapter_params, self.config.grad_clip)

        # Apply the warmup schedule to the LR before stepping.
        lr = self.config.lr_at(self.state.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.step()

        loss_val = float(loss.item())
        self.state.step += 1
        self.state.losses.append(loss_val)
        return loss_val

    # ---------------------------------------------------------------- #
    # Full training loop over records                                  #
    # ---------------------------------------------------------------- #
    def train(self, records) -> Dict[str, object]:
        """Train for config.epochs over the records; return a run summary."""
        records = list(records)
        if not records:
            raise ValueError("cannot train on an empty record set")

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            for batch in iter_batches(
                records,
                self.tokenizer,
                self.config.max_seq_len,
                self.config.batch_size,
            ):
                self.step(batch)

        first = self.state.losses[0] if self.state.losses else None
        last = self.state.last_loss
        return {
            "steps": self.state.step,
            "epochs": self.config.epochs,
            "initial_loss": first,
            "final_loss": last,
            "loss_improved": (
                bool(last < first) if (first is not None and last is not None) else None
            ),
            "loss_history": list(self.state.losses),
            **self.param_stats(),
        }

    # ---------------------------------------------------------------- #
    # Evaluation over a held-out record set                            #
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def evaluate(self, records) -> EvalResult:
        """Compute perplexity + token accuracy on records (no gradient)."""
        records = list(records)
        if not records:
            raise ValueError("cannot evaluate on an empty record set")
        self.model.eval()
        evaluator = Evaluator()
        for batch in iter_batches(
            records,
            self.tokenizer,
            self.config.max_seq_len,
            self.config.batch_size,
        ):
            logits = self.model(batch["input_ids"])
            # Shift so position t scores token t+1, matching the training loss.
            shift_logits = logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            evaluator.update(shift_logits, shift_labels)
        return evaluator.result()
