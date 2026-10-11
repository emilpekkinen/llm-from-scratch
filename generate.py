"""Sample text from a trained checkpoint."""
import argparse
import os

import numpy as np

from model import GPT
from tokenizer import BPETokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    tok = BPETokenizer.load(os.path.join(args.out_dir, "tokenizer.json"))
    model, extra = GPT.load(os.path.join(args.out_dir, "ckpt.npz"))
    print(f"loaded checkpoint from step {extra.get('step')} (val loss {extra.get('val_loss', float('nan')):.4f})\n")
    ids = model.generate(tok.encode(args.prompt), args.max_new_tokens, args.temperature,
                         args.top_k, rng=np.random.default_rng(args.seed))
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
