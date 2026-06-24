"""microgpt — live demo.

Train a small GPT from scratch on a corpus, generate text, and *see* the
attention maps. Everything — the BPE tokenizer, the transformer, the training
loop — is hand-written (model.py, bpe.py, train.py). No Hugging Face.

On a GPU Studio this trains a coherent tiny model in a couple of minutes; on
CPU, keep the steps low for the demo.
"""

from __future__ import annotations

import io
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from bpe import BPE
from model import GPT, GPTConfig

STATE = {"model": None, "tok": None, "device": "cuda" if torch.cuda.is_available() else "cpu"}


def _get_batch(data, block, batch, device):
    ix = torch.randint(len(data) - block, (batch,))
    x = torch.stack([data[i : i + block] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def train_live(corpus, steps, progress=gr.Progress()):
    device = STATE["device"]
    if len(corpus.strip()) < 200:
        return "Please paste at least a few paragraphs of text to train on.", None

    tok = BPE()
    tok.train(corpus, vocab_size=512)
    ids = torch.tensor(tok.encode(corpus), dtype=torch.long)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    model.train()
    steps = int(steps)
    for step in progress.tqdm(range(steps), desc="training"):
        x, y = _get_batch(ids, cfg.block_size, 16, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 50) == 0:
            losses.append(float(loss))

    STATE["model"], STATE["tok"] = model, tok

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(losses, color="#7c5cff", lw=2)
    ax.set_title("Training loss"); ax.set_xlabel("checkpoint"); ax.grid(alpha=0.2)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=100); plt.close(fig); buf.seek(0)
    msg = (
        f"Trained a {model.num_params()/1e6:.2f}M-param GPT on {device.upper()} · "
        f"vocab {tok.vocab_size} · final loss {losses[-1]:.3f}. Now generate below."
    )
    return msg, Image.open(buf)


def generate(prompt, max_tokens, temperature):
    if STATE["model"] is None:
        return "Train a model first (top section).", None
    model, tok, device = STATE["model"], STATE["tok"], STATE["device"]
    model.eval()
    if prompt.strip():
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    else:
        ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=int(max_tokens), temperature=float(temperature), top_k=40)
    text = tok.decode(out[0].tolist())

    # attention map from the last layer, head 0, on the prompt
    attn_img = None
    block = model.blocks[-1].attn
    if block._last_attn is not None:
        a = block._last_attn[0, 0].cpu().numpy()  # (T, T)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(a, cmap="magma")
        ax.set_title("Last-layer attention (head 0)")
        ax.set_xlabel("key position"); ax.set_ylabel("query position")
        buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=100); plt.close(fig); buf.seek(0)
        attn_img = Image.open(buf)
    return text, attn_img


with gr.Blocks(title="microgpt", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 microgpt — a GPT built & trained from scratch\n"
        "A decoder-only transformer and a BPE tokenizer, **written by hand in "
        "pure PyTorch** (no Hugging Face). Train it live on any text, generate "
        "samples, and inspect the attention maps. "
        f"_Running on **{STATE['device'].upper()}**._"
    )
    with gr.Tab("1 · Train"):
        corpus = gr.Textbox(label="Training corpus", lines=10, placeholder="Paste a few paragraphs (or a whole book) here…")
        steps = gr.Slider(100, 3000, value=600, step=100, label="Training steps")
        train_btn = gr.Button("Train from scratch", variant="primary")
        train_msg = gr.Markdown()
        loss_plot = gr.Image(label="Loss curve")
        train_btn.click(train_live, [corpus, steps], [train_msg, loss_plot])
    with gr.Tab("2 · Generate"):
        prompt = gr.Textbox(label="Prompt (optional)", value="")
        with gr.Row():
            max_tokens = gr.Slider(20, 500, value=200, step=20, label="Max new tokens")
            temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
        gen_btn = gr.Button("Generate", variant="primary")
        gen_out = gr.Textbox(label="Generated text", lines=10)
        attn = gr.Image(label="Attention map")
        gen_btn.click(generate, [prompt, max_tokens, temperature], [gen_out, attn])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
