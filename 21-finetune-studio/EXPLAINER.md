# EXPLAINER — the math behind finetune-studio

This document explains, in depth, the two pieces of math the project implements
from scratch: **LoRA adaptation** and the **evaluation metrics**.

---

## 1. LoRA: Low-Rank Adaptation

### 1.1 The problem

A pretrained linear layer holds a weight matrix `W ∈ R^{out×in}`. Full
fine-tuning learns a dense update `ΔW` of the same shape and adds it: `W + ΔW`.
For a 4096×4096 attention projection that is ~16.7M trainable parameters **per
layer** — expensive to train, expensive to store one copy per task.

### 1.2 The low-rank hypothesis

LoRA assumes the *update* a model needs to adapt to a downstream task has low
intrinsic rank. If `ΔW` is approximately rank `r` with `r ≪ min(in, out)`, we can
factor it as a product of two thin matrices:

```
ΔW  =  B · A        A ∈ R^{r×in},   B ∈ R^{out×r}
```

`A` projects the `in`-dim input down to an `r`-dim bottleneck; `B` projects back
up to `out`. We then scale by `α/r` (see §1.4) and add to the frozen base:

```
W_adapted  =  W  +  (α / r) · B · A
```

Trainable parameters drop from `out×in` to `r·(in + out)`. For the 4096×4096
example at `r = 8`: `8·(4096+4096) = 65,536` — a ~256× reduction.

### 1.3 The forward pass

We never materialize the dense `ΔW` during training. Instead, for an input
`x ∈ R^{...×in}`, we route it through the bottleneck:

```
y  =  x · Wᵀ  +  (α/r) · ( dropout(x) · Aᵀ ) · Bᵀ   (+ bias)
       └ base ┘   └──────── adapter path ─────────┘
```

In code (`core/lora.py`), the base path is the original `nn.Linear`; the adapter
path is two `F.linear` calls (`x → r-dim → out-dim`). Dropout is applied to the
**adapter input only**, so it regularizes the learned update without ever
perturbing the frozen base path.

### 1.4 Why scale by α/r?

The factor `γ = α/r` decouples the *learning-rate-like* magnitude of the update
from the rank you happen to choose. If you increase `r` to give the adapter more
capacity, dividing by `r` keeps the typical magnitude of `γ·B·A` roughly
constant, so you don't have to re-tune the learning rate every time you change
the rank. `α` is the single knob controlling how strongly the adapter speaks.

### 1.5 Identity initialization (B = 0)

We initialize `A` with Kaiming-uniform (keeps the bottleneck activations
well-scaled) and **`B = 0`**. Then at step 0:

```
γ · B · A  =  γ · 0 · A  =  0     ⇒     y = x·Wᵀ + bias  =  base output
```

The adapted model is **identical** to the pretrained model at initialization.
Training therefore starts exactly from the pretrained solution and moves away
from it gradually — no destructive shock from a randomly-initialized adapter, and
the `α/r` scale can't blow up the outputs before any learning has happened. Test
(2) asserts `lora(x) == base(x)` at init to `1e-6`.

### 1.6 Merging for inference

Because the adapter is *linear*, it distributes over the base weight. After
training we can fold it in once:

```
W_merged  =  W  +  (α/r) · B · A
```

Now a single plain `nn.Linear` with weight `W_merged` reproduces the full LoRA
forward — **zero extra latency or memory at inference**. `merge()` does exactly
this in place (and is guarded so a double-merge can't corrupt the weight);
`unmerge()` subtracts it back to recover the trainable form. Test (4) proves the
merged layer's output matches the un-merged LoRA forward to `1e-5`.

### 1.7 Training only the adapter

`inject_lora()` walks the model, replaces every targeted `nn.Linear` (matched by
name substring, e.g. `q_proj`, `v_proj`) with a `LoRALinear`, and sets
`requires_grad=False` on the base weight/bias. The trainer then collects **only**
the `A`/`B` parameters and hands those to the optimizer, so the frozen base never
moves. A test snapshots a base weight and confirms it is unchanged after a step.

---

## 2. Evaluation metrics

### 2.1 Perplexity

A language model defines `P(token | context)`. Its fit to held-out text is the
average **negative log-likelihood** per token:

```
NLL  =  -(1/N) · Σ_t  log P(x_t | x_<t)
```

**Perplexity** is the exponential of that mean:

```
PPL  =  exp(NLL)
```

Interpretation: PPL is the *effective branching factor* — the model was, on
average, as uncertain as if choosing uniformly among `PPL` equally-likely tokens.
A perfect model scores 1; a uniform model over a `V`-token vocabulary scores `V`.

Implementation (`core/eval.py`):

1. `log_softmax(logits)` along the vocab axis — numerically stable; never forms
   the raw softmax then logs it (which overflows/underflows).
2. `gather` the log-prob of the **gold** token at each position.
3. Mask out positions where the label is `IGNORE_INDEX` (prompt tokens, padding,
   and the dangling shift position) so they contribute nothing.
4. Average the surviving log-probs, negate, exponentiate.

**Worked example (test 5).** Vocab size 2, two positions:

- Position 0: logits `[0, 0]` → softmax `[0.5, 0.5]`; gold `0` → `log 0.5`.
- Position 1: logits `[ln 3, 0]` → softmax `[0.75, 0.25]`; gold `1` → `log 0.25`.

```
NLL = -½(ln 0.5 + ln 0.25)
PPL = exp(NLL) = 1/√(0.5·0.25) = √8 ≈ 2.828
```

The test pins the computed perplexity to `√8`.

The streaming `Evaluator` accumulates `Σ NLL` and a token count, so the
corpus-level perplexity is `exp(ΣNLL / Σtokens)` — the **token-weighted** mean,
which is the correct definition for ragged batches (averaging per-batch
perplexities would be wrong).

### 2.2 Token accuracy & exact match

For tasks with discrete answers we also report **token-level accuracy**: over the
supervised (non-masked) positions, the fraction where `argmax(logits)` equals the
gold token. Test (6) constructs four positions with three correct and asserts
`0.75`.

`sequence_exact_match` reports the fraction of whole predictions that match their
reference after light normalization (lower-case, collapse whitespace) — the
standard exact-match metric used in QA evaluation.

### 2.3 Why these are computed on shifted logits

A causal LM at position `t` predicts token `t+1`. So both the training loss and
the evaluator align `logits[:, :-1]` with `labels[:, 1:]` (drop the last logit
and first label, which have no partner). This shift is the crux of next-token
modeling and is applied consistently in `causal_lm_loss` and `Evaluator`.
