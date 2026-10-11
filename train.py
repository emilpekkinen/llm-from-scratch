"""Train the NumPy GPT on a text file (defaults to Tiny Shakespeare)."""
import argparse
import math
import os
import time
import urllib.request

import numpy as np

from model import GPT, AdamW, GPTConfig, clip_grad_norm
from tokenizer import BPETokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt", help="training text file (downloaded if missing)")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--min_lr", type=float, default=2e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def load_data(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(args.data):
        os.makedirs(os.path.dirname(args.data) or ".", exist_ok=True)
        print(f"downloading Tiny Shakespeare -> {args.data}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, args.data)
    with open(args.data, encoding="utf-8") as f:
        text = f.read()

    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    ids_path = os.path.join(args.out_dir, f"tokens_{args.vocab_size}.npy")
    if os.path.exists(tok_path) and os.path.exists(ids_path):
        tok = BPETokenizer.load(tok_path)
        ids = np.load(ids_path)
    else:
        print(f"training BPE tokenizer (vocab {args.vocab_size}) on {len(text):,} chars...")
        t0 = time.time()
        tok = BPETokenizer()
        tok.train(text, args.vocab_size, verbose=True)
        tok.save(tok_path)
        ids = np.array(tok.encode(text), dtype=np.int32)
        np.save(ids_path, ids)
        print(f"tokenizer done in {time.time() - t0:.1f}s")
    print(f"{len(text):,} chars -> {len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token)")
    n = int(0.9 * len(ids))
    return tok, ids[:n], ids[n:]


def get_batch(data, block_size, batch_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def lr_at(it, args):
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def estimate_loss(model, train, val, args, rng):
    model.eval()  # left in eval mode; the training loop switches back after sampling
    out = {}
    for name, data in (("train", train), ("val", val)):
        losses = [model.forward(*get_batch(data, args.block_size, args.batch_size, rng))[1]
                  for _ in range(args.eval_iters)]
        out[name] = float(np.mean(losses))
    return out


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    tok, train, val = load_data(args)

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    dropout=args.dropout)
    model = GPT(cfg, seed=args.seed)
    params = list(model.parameters().values())
    opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"model: {model.num_params():,} parameters | {cfg}")
    print(f"initial loss should be ~ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.2f}")

    ckpt = os.path.join(args.out_dir, "ckpt.npz")
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, train, val, args, rng)
            print(f"--- step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                model.save(ckpt, step=it, val_loss=best_val)
            sample = model.generate(tok.encode("\n"), 80, temperature=0.8, top_k=40, rng=rng)
            print(tok.decode(sample).strip().replace("\n", " | "))
            if it == args.max_iters:
                break
            model.train()
            t0 = time.time()

        lr = lr_at(it, args)
        x, y = get_batch(train, args.block_size, args.batch_size, rng)
        model.zero_grad()
        _, loss = model.forward(x, y)
        model.backward()
        gnorm = clip_grad_norm(params, args.grad_clip)
        opt.step(lr)

        if it % 10 == 9:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it:5d} | loss {loss:.4f} | lr {lr:.2e} | gnorm {gnorm:.2f} | {dt * 100:.0f} ms/iter")

    print(f"done. best val loss {best_val:.4f}, checkpoint at {ckpt}")


if __name__ == "__main__":
    main()
