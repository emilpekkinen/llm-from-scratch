"""Verify the hand-written backward pass against finite differences (float64)."""
import numpy as np

from model import GPT, GPTConfig


def check(dropout):
    cfg = GPTConfig(vocab_size=37, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=dropout, dtype="float64")
    model = GPT(cfg, seed=1)
    model.train()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cfg.vocab_size, (3, 8))
    tgt = rng.integers(0, cfg.vocab_size, (3, 8))

    def loss_fn():
        # Same dropout masks on every forward pass, so finite differences are meaningful
        model.set_dropout_rng(np.random.default_rng(7))
        return model.forward(idx, tgt)

    model.zero_grad()
    loss_fn()
    model.backward()

    eps, worst = 1e-5, 0.0
    for name, p in model.parameters().items():
        flat, grad = p.data.reshape(-1), p.grad.reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            _, lp = loss_fn()
            flat[i] = old - eps
            _, lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - grad[i]) / max(1e-8, abs(num) + abs(grad[i]))
            worst = max(worst, rel)
            if rel > 1e-4:
                print(f"MISMATCH {name}[{i}]: analytic={grad[i]:.6e} numeric={num:.6e} rel={rel:.2e}")
    print(f"dropout={dropout}: checked {len(model.parameters())} tensors, worst relative error {worst:.2e}")
    assert worst < 1e-4, "gradient check failed"


def main():
    check(0.0)
    check(0.2)
    print("gradient check passed")


if __name__ == "__main__":
    main()
