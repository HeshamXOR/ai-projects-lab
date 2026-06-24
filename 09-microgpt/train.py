"""Train microgpt on a text corpus. Designed for a single GPU on Lightning AI
(or CPU for a tiny config).

    python train.py --data data/input.txt --steps 2000 --vocab 512

Saves model + tokenizer to out/. Prints a decreasing loss curve to stdout and
writes loss_curve.txt so you can chart it for the README.
"""

from __future__ import annotations

import argparse
import os

import torch

from bpe import BPE
from model import GPT, GPTConfig


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    os.makedirs(args.out, exist_ok=True)

    text = open(args.data, "r", encoding="utf-8").read()
    print(f"corpus: {len(text)} chars")

    # 1) train the BPE tokenizer from scratch on this corpus
    tok = BPE()
    tok.train(text, vocab_size=args.vocab)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    print(f"tokenizer: {tok.vocab_size} tokens")

    # 2) encode the whole corpus
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_data, val_data = ids[:n], ids[n:]

    # 3) build the model
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block)
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params()/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4) train
    losses = []
    model.train()
    for step in range(args.steps):
        x, y = get_batch(train_data, args.block, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            losses.append((step, float(loss)))
            print(f"step {step:5d} | loss {float(loss):.4f}")

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "loss_curve.txt"), "w") as f:
        for s, l in losses:
            f.write(f"{s}\t{l}\n")

    # 5) sample
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=300, temperature=0.8, top_k=40)
    print("\n=== sample ===")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
