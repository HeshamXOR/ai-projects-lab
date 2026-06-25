# finetune-studio

## What I implemented from scratch

A complete LoRA fine-tuning, evaluation, and serving pipeline — the hard parts
written by hand in PyTorch/NumPy rather than pulled from `peft`/`trl`:

- **LoRA adapter math** (`core/lora.py`). A `LoRALinear` that wraps an
  `nn.Linear` and adds the low-rank update `W + (alpha/r) * B @ A`. Includes:
  - forward pass with input **dropout** on the adapter path only,
  - **zero-init of B** so the adapter starts as identity (output == base at init),
  - `merge()` / `unmerge()` that fold the adapter into the base weight for
    zero-overhead inference (and are idempotent-guarded),
  - `inject_lora()` that walks a model, replaces target linear layers in place,
    and **freezes the base** so only adapters train.
- **Training config + dataset pipeline** (`core/config.py`, `core/data.py`).
  A validated `TrainingConfig` dataclass (lr, r, alpha, dropout, epochs,
  batch_size, warmup, grad-clip, ...) with a derived `scaling = alpha/r` and a
  warmup LR schedule; an **Alpaca-style prompt formatter**; a dependency-free
  **char-level tokenizer** with an HF-compatible interface; and tokenization with
  **prompt-token label masking** (`IGNORE_INDEX`) so loss is computed only over
  the response.
- **Eval harness** (`core/eval.py`). **Perplexity** = `exp(mean NLL)` computed
  from a numerically stable log-softmax and gathered gold log-probs, plus
  **token-level accuracy** and **sequence exact-match** — all hand-implemented and
  pinned to hand-computed values in the tests. A streaming `Evaluator` gives exact
  corpus-level perplexity over ragged batches.
- **Trainer** (`core/trainer.py`). Injects adapters, hands **only adapter params**
  to Adam, runs the shifted next-token cross-entropy loop with gradient clipping,
  and exposes `step()` / `train()` / `evaluate()`.
- **Pluggable tiny base LM** (`core/model.py`). A real but miniature causal
  transformer (`TinyCausalLM`) with named `q_proj`/`v_proj` linears so injection
  has real targets — runs with **zero downloads**. The `CausalLM` Protocol is the
  dependency-injection seam for swapping in a real Hugging Face model.

## Run it

```bash
pip install -r requirements.txt        # torch, fastapi, uvicorn, pydantic, numpy, pytest

# Run the tests that prove the core math:
pytest -q

# Start the service:
uvicorn app:app --reload
# or: python app.py
```

Docker:

```bash
docker build -t finetune-studio .
docker run -p 8000:8000 finetune-studio
```

## API

All endpoints return structured JSON. The base LLM is the deterministic
`TinyCausalLM` by default — no weights are downloaded.

| Method | Path        | Purpose |
|--------|-------------|---------|
| GET    | `/health`   | Liveness + which base model is loaded. |
| POST   | `/finetune` | Run a tiny LoRA run on supplied records; returns param accounting, loss history, and perplexity before/after. Caches the trained adapter under a `session_id`. |
| POST   | `/eval`     | Perplexity + token accuracy on records, optionally using a trained `session_id`. |
| POST   | `/generate` | Greedy/sampled generation from base (or a trained adapter session). |

### Example: fine-tune then generate

```bash
# 1) Fine-tune on a couple of records.
curl -s localhost:8000/finetune -H 'content-type: application/json' -d '{
  "records": [
    {"instruction": "Say hi", "output": "hello"},
    {"instruction": "Add", "input": "2+2", "output": "4"}
  ],
  "config": {"epochs": 5, "r": 4, "alpha": 8}
}'
# -> {"session_id": "...", "param_stats": {...}, "perplexity_before": ..., "perplexity_after": ...}

# 2) Generate using that adapter.
curl -s localhost:8000/generate -H 'content-type: application/json' -d '{
  "instruction": "Say hi",
  "session_id": "<session_id from above>"
}'
```

See **EXPLAINER.md** for the full derivation of the LoRA math and eval metrics.

## Plugging in a real model

`core/model.py` defines a `CausalLM` Protocol: any module with
`forward(input_ids) -> logits [B, T, V]` and a `vocab_size` works. Wrap a Hugging
Face `AutoModelForCausalLM`, swap `Studio.fresh_model` / `Studio.tokenizer` in
`app.py`, and `inject_lora(target_modules=["q_proj", "v_proj"])` finds the real
attention projections unchanged.
