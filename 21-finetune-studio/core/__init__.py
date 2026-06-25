"""From-scratch LoRA fine-tuning, eval, and serving core.

Public surface re-exported for convenient imports:

    from core import LoRALinear, TrainingConfig, LoRATrainer, perplexity, ...
"""

from .config import TrainingConfig
from .data import (
    IGNORE_INDEX,
    AlpacaFormatter,
    CharTokenizer,
    FormattedExample,
    Tokenizer,
    build_training_batch,
    encode_with_mask,
    format_example,
    iter_batches,
)
from .eval import (
    EvalResult,
    Evaluator,
    gather_token_log_probs,
    mean_negative_log_likelihood,
    perplexity,
    sequence_exact_match,
    token_accuracy,
)
from .lora import (
    LoRALinear,
    inject_lora,
    iter_lora_layers,
    merge_all,
    trainable_adapter_parameters,
)
from .model import CausalLM, TinyCausalLM
from .trainer import LoRATrainer, TrainState, causal_lm_loss

__all__ = [
    # config
    "TrainingConfig",
    # data
    "IGNORE_INDEX",
    "AlpacaFormatter",
    "CharTokenizer",
    "FormattedExample",
    "Tokenizer",
    "build_training_batch",
    "encode_with_mask",
    "format_example",
    "iter_batches",
    # eval
    "EvalResult",
    "Evaluator",
    "gather_token_log_probs",
    "mean_negative_log_likelihood",
    "perplexity",
    "sequence_exact_match",
    "token_accuracy",
    # lora
    "LoRALinear",
    "inject_lora",
    "iter_lora_layers",
    "merge_all",
    "trainable_adapter_parameters",
    # model
    "CausalLM",
    "TinyCausalLM",
    # trainer
    "LoRATrainer",
    "TrainState",
    "causal_lm_loss",
]
