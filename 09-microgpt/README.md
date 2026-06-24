# 🧠 microgpt — a GPT built and trained from scratch

A decoder-only transformer **and** a BPE tokenizer, written by hand in pure PyTorch — **no Hugging Face, no `nn.Transformer`**. Train it live on any text, generate samples, and inspect the attention maps.

![preview](preview.gif)
<!-- Record the train→generate flow on Lightning (GPU) and save as preview.gif. -->

## What I implemented from scratch

- **BPE tokenizer** — trains merges from raw bytes; encodes/decodes with exact round-trip (`bpe.py`)
- **Transformer** — causal multi-head self-attention, pre-norm residual blocks, weight-tied head (`model.py`)
- **Training loop + sampling** (`train.py`, `model.generate`)

See [EXPLAINER.md](EXPLAINER.md) for how each piece works.

## Run it

**Demo (Gradio):**
```bash
pip install -r requirements.txt
python app.py        # Train tab → paste text → train; Generate tab → sample + attention map
```

**CLI training:**
```bash
python train.py --data data/input.txt --steps 2000 --vocab 512
# saves out/model.pt, out/tokenizer.json, out/loss_curve.txt
```

A GPU Studio (L4) trains a coherent tiny model in a couple of minutes. CPU works for small step counts.

## Verify

```bash
pytest -q     # BPE round-trip + compression; model forward/backward/generate; weight tying
```

## Why it's here

This is the portfolio's deep-learning centerpiece: it shows the transformer architecture isn't a black box to me — I can build the tokenizer, the attention, the training loop, and the sampler from the math up.

## Limitations

- Small by design (trains on free hardware); byte-level BPE without GPT-2's regex pre-tokenization; no KV-cache in generation. All documented in the EXPLAINER.
