"""A tiny, deterministic causal language model — the pluggable base.

The WHY: the whole point of this studio is the LoRA / training / eval machinery,
NOT a 7B-parameter download. So we ship a genuinely-runnable miniature causal LM
built from real components (embedding -> a couple of tiny transformer-ish blocks
with named q_proj/v_proj linears so injection has something to target -> tied
output head). It is small enough to train on CPU in a test, but it is a *real*
nn.Module producing [B, T, V] logits, so every downstream piece (LoRA injection,
loss, perplexity, generation) exercises the same code paths a production model
would.

The CausalLM Protocol at the bottom is the contract the FastAPI service codes
against. To plug a real Hugging Face model, wrap it so .forward(input_ids) ->
logits [B, T, V] and expose .config.vocab_size; nothing else in the pipeline
changes. That is the dependency-injection seam.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F


@runtime_checkable
class CausalLM(Protocol):
    """Contract every base model must satisfy to be fine-tunable here."""

    vocab_size: int

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] -> logits [B, T, vocab_size]."""
        ...


class TinyAttention(nn.Module):
    """Single-head causal self-attention with explicitly named projections.

    The projections are deliberately named q_proj / k_proj / v_proj / o_proj so
    that inject_lora(target_modules=["q_proj", "v_proj"]) finds real layers, just
    like it would in a Llama/GPT-style model.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)        # [B, T, T]
        # Causal mask: position i may attend to j <= i.
        causal = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                          # [B, T, d]
        return self.o_proj(out)


class TinyBlock(nn.Module):
    """Pre-norm transformer block: attention + MLP, both residual."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyCausalLM(nn.Module):
    """A small but complete causal language model.

    Token embedding + learned positional embedding -> N TinyBlocks -> final
    norm -> output head (weight-tied to the embedding, the standard trick that
    halves params and usually helps). Returns logits [B, T, V].
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 512,
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.vocab_size = int(vocab_size)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TinyBlock(d_model, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying.
        self.head.weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max {self.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                                     # [B, T, V]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        eos_id: Optional[int] = None,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Autoregressively extend input_ids by up to max_new_tokens.

        Greedy by default (deterministic, good for tests/demo). With greedy=False
        it samples from the temperature-scaled softmax. Stops early on eos_id.
        """
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            window = ids[:, -self.max_seq_len :]
            logits = self.forward(window)[:, -1, :]            # [B, V]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return ids
