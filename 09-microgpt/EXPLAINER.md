# EXPLAINER — microgpt: a GPT built and trained from scratch

## What I implemented from scratch

- A **BPE tokenizer** (the GPT-2/Llama tokenization algorithm) — trains merges from raw bytes, encodes, decodes, round-trips exactly. `bpe.py`
- A **decoder-only transformer** in pure PyTorch — causal multi-head self-attention, residual+prenorm blocks, token/positional embeddings, weight-tied head — no Hugging Face, no `nn.Transformer`. `model.py`
- The **training loop** and autoregressive **sampling**. `train.py`, `model.generate`

## How BPE works (`bpe.py`)

Byte-Pair Encoding starts from the 256 raw byte values and repeatedly finds the **most frequent adjacent pair** of tokens, merges it into a new token, and records the merge. After N merges you have a 256+N vocabulary where common sequences ("the ", "ing") are single tokens.

- **Training**: count adjacent pairs → merge the top pair → repeat. Each merge is stored as `(a, b) -> new_id` and the new token's bytes are `vocab[a] + vocab[b]`.
- **Encoding**: greedily apply the learned merges in the order they were learned (earliest merge first), which reproduces the training-time segmentation.
- **Decoding**: concatenate each token's bytes and UTF-8 decode — so it round-trips *exactly*, including Unicode and emoji (because we operate on bytes, not characters).

## How the transformer works (`model.py`)

Reading top to bottom, a sequence of token ids becomes a next-token prediction:

1. **Embeddings**: each token id → a learned vector; add a learned **positional** embedding so the model knows order.
2. **Causal self-attention**: project to Q, K, V for every head; compute `softmax(QKᵀ/√d)·V` with a **lower-triangular mask** so a token can only attend to itself and earlier tokens (a language model must not see the future). This is where the model decides *which earlier tokens matter* for predicting the next one.
3. **MLP block**: a 2-layer feed-forward net with GELU widens then narrows the representation.
4. **Residuals + LayerNorm** (pre-norm) around both sub-layers, stacked `n_layer` times — this is what makes deep transformers trainable.
5. **Output head**: project back to vocabulary logits. The head **shares weights** with the input embedding (weight tying) — a standard trick that saves parameters and improves quality.
6. **Generation**: take the last position's logits, apply temperature + top-k, sample, append, repeat.

## Training (`train.py`)

Trains the BPE tokenizer on the corpus, encodes it, then optimizes next-token cross-entropy with AdamW. On a GPU Studio a tiny model becomes coherent in minutes; the loss curve and a text sample are printed and saved.

## Proof it works

`tests/test_microgpt.py` (run `pytest`):
- BPE **round-trips exactly** on ASCII and on Unicode/emoji.
- BPE **actually compresses** (repeated patterns merge: encoded length < half the raw byte length).
- The model's forward pass produces correct logit shapes, a positive loss, and a working backward pass; `generate` extends the sequence; weight tying is wired correctly.

The honest end-to-end proof is the **training loss curve** — run `train.py` (or the app's Train tab) on Lightning and you'll see it fall and the samples become text-like. A model this small on a small corpus won't be GPT-4; the point is that the *mechanism* is real and mine.

## Limitations

- Small by design (a few hundred K to a few M params) so it trains on free Lightning hardware.
- Byte-level BPE without the regex pre-tokenization GPT-2 uses (kept simple for readability).
- No KV-cache in generation (recomputes context each step) — fine at this scale, a known optimization to add.
