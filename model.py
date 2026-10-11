"""A GPT-style decoder-only transformer in pure NumPy, with hand-written backprop.

Every layer has a forward() that caches what it needs and a backward() that
takes dL/d(output), accumulates parameter gradients into Param.grad, and
returns dL/d(input).
"""
import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class GPTConfig:
    vocab_size: int = 512
    block_size: int = 128  # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    dtype: str = "float32"


class Param:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)


# ----------------------------------------------------------------------------- layers

class Linear:
    def __init__(self, n_in, n_out, rng, std=0.02, bias=True):
        self.W = Param(rng.normal(0.0, std, (n_in, n_out)))
        self.b = Param(np.zeros(n_out)) if bias else None

    def params(self):
        return {"W": self.W} | ({"b": self.b} if self.b is not None else {})

    def forward(self, x):
        self.x = x
        y = x @ self.W.data
        return y + self.b.data if self.b is not None else y

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.W.grad += x2.T @ dy2
        if self.b is not None:
            self.b.grad += dy2.sum(0)
        return dy @ self.W.data.T


class LayerNorm:
    def __init__(self, n, eps=1e-5):
        self.g = Param(np.ones(n))
        self.b = Param(np.zeros(n))
        self.eps = eps

    def params(self):
        return {"g": self.g, "b": self.b}

    def forward(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.rstd
        return self.xhat * self.g.data + self.b.data

    def backward(self, dy):
        n = dy.shape[-1]
        self.g.grad += (dy * self.xhat).reshape(-1, n).sum(0)
        self.b.grad += dy.reshape(-1, n).sum(0)
        dxhat = dy * self.g.data
        return self.rstd * (
            dxhat
            - dxhat.mean(-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(-1, keepdims=True)
        )


class GELU:
    """tanh approximation, as in GPT-2."""
    C = math.sqrt(2.0 / math.pi)

    def params(self):
        return {}

    def forward(self, x):
        self.x = x
        self.t = np.tanh(self.C * (x + 0.044715 * x**3))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dy):
        x, t = self.x, self.t
        du = self.C * (1.0 + 3 * 0.044715 * x**2)
        return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du)


class Dropout:
    """Inverted dropout: zero a fraction p of activations during training and
    scale the rest by 1/(1-p), so nothing needs to change at inference time."""

    def __init__(self, p):
        self.p = p
        self.training = False
        self.rng = np.random.default_rng()
        self.mask = None

    def params(self):
        return {}

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dy):
        return dy if self.mask is None else dy * self.mask


class CausalSelfAttention:
    def __init__(self, cfg, rng):
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd, rng)
        # Residual projections get scaled-down init (GPT-2)
        self.c_proj = Linear(cfg.n_embd, cfg.n_embd, rng, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.mask = np.tril(np.ones((cfg.block_size, cfg.block_size), dtype=bool))
        self.attn_drop = Dropout(cfg.dropout)
        self.resid_drop = Dropout(cfg.dropout)

    def params(self):
        return {"c_attn": self.c_attn, "c_proj": self.c_proj}

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, C // self.n_head
        qkv = self.c_attn.forward(x)
        # (B, T, C) -> (B, H, T, D)
        q, k, v = (a.reshape(B, T, H, D).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, axis=-1))
        self.scale = 1.0 / math.sqrt(D)
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        att = np.where(self.mask[:T, :T], att, -np.inf)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        att_d = self.attn_drop.forward(att)
        y = att_d @ v
        self.q, self.k, self.v, self.att, self.att_d = q, k, v, att, att_d
        return self.resid_drop.forward(self.c_proj.forward(y.transpose(0, 2, 1, 3).reshape(B, T, C)))

    def backward(self, dout):
        B, H, T, D = self.q.shape
        dy = self.c_proj.backward(self.resid_drop.backward(dout)).reshape(B, T, H, D).transpose(0, 2, 1, 3)
        datt = self.attn_drop.backward(dy @ self.v.transpose(0, 1, 3, 2))
        dv = self.att_d.transpose(0, 1, 3, 2) @ dy
        # Softmax backward; masked positions have att == 0 so they receive no gradient
        dscores = self.att * (datt - (datt * self.att).sum(-1, keepdims=True)) * self.scale
        dq = dscores @ self.k
        dk = dscores.transpose(0, 1, 3, 2) @ self.q
        dqkv = np.concatenate(
            [a.transpose(0, 2, 1, 3).reshape(B, T, H * D) for a in (dq, dk, dv)], axis=-1
        )
        return self.c_attn.backward(dqkv)


class MLP:
    def __init__(self, cfg, rng):
        self.c_fc = Linear(cfg.n_embd, 4 * cfg.n_embd, rng)
        self.gelu = GELU()
        self.c_proj = Linear(4 * cfg.n_embd, cfg.n_embd, rng, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self.drop = Dropout(cfg.dropout)

    def params(self):
        return {"c_fc": self.c_fc, "c_proj": self.c_proj}

    def forward(self, x):
        return self.drop.forward(self.c_proj.forward(self.gelu.forward(self.c_fc.forward(x))))

    def backward(self, dy):
        return self.c_fc.backward(self.gelu.backward(self.c_proj.backward(self.drop.backward(dy))))


class Block:
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg, rng):
        self.ln_1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, rng)
        self.ln_2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg, rng)

    def params(self):
        return {"ln_1": self.ln_1, "attn": self.attn, "ln_2": self.ln_2, "mlp": self.mlp}

    def forward(self, x):
        x = x + self.attn.forward(self.ln_1.forward(x))
        return x + self.mlp.forward(self.ln_2.forward(x))

    def backward(self, dx):
        dx = dx + self.ln_2.backward(self.mlp.backward(dx))
        return dx + self.ln_1.backward(self.attn.backward(dx))


# ----------------------------------------------------------------------------- model

class GPT:
    def __init__(self, cfg: GPTConfig, seed=0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.wte = Param(rng.normal(0.0, 0.02, (cfg.vocab_size, cfg.n_embd)))
        self.wpe = Param(rng.normal(0.0, 0.01, (cfg.block_size, cfg.n_embd)))
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)
        self.lm_head = Linear(cfg.n_embd, cfg.vocab_size, rng, bias=False)
        self.drop = Dropout(cfg.dropout)
        for p in self.parameters().values():
            p.data = p.data.astype(cfg.dtype)
            p.grad = np.zeros_like(p.data)
        self.set_dropout_rng(np.random.default_rng(seed + 1))

    def dropouts(self):
        out = [self.drop]
        for b in self.blocks:
            out += [b.attn.attn_drop, b.attn.resid_drop, b.mlp.drop]
        return out

    def train(self, mode=True):
        """Enable dropout (training) or disable it (evaluation / generation)."""
        for d in self.dropouts():
            d.training = mode

    def eval(self):
        self.train(False)

    def set_dropout_rng(self, rng):
        for d in self.dropouts():
            d.rng = rng

    def parameters(self):
        """Flat {dotted.name: Param} dict over the whole model."""
        out = {}

        def walk(prefix, obj):
            if isinstance(obj, Param):
                out[prefix] = obj
            else:
                for k, v in obj.params().items():
                    walk(f"{prefix}.{k}" if prefix else k, v)

        walk("wte", self.wte)
        walk("wpe", self.wpe)
        for i, b in enumerate(self.blocks):
            walk(f"h.{i}", b)
        walk("ln_f", self.ln_f)
        walk("lm_head", self.lm_head)
        return out

    def num_params(self):
        return sum(p.data.size for p in self.parameters().values())

    def zero_grad(self):
        for p in self.parameters().values():
            p.grad.fill(0)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        self.idx = idx
        x = self.drop.forward(self.wte.data[idx] + self.wpe.data[:T])
        for block in self.blocks:
            x = block.forward(x)
        logits = self.lm_head.forward(self.ln_f.forward(x))
        if targets is None:
            return logits, None
        # Cross-entropy, numerically stable
        flat = logits.reshape(-1, logits.shape[-1])
        flat = flat - flat.max(-1, keepdims=True)
        probs = np.exp(flat)
        probs /= probs.sum(-1, keepdims=True)
        t = targets.reshape(-1)
        loss = -np.log(probs[np.arange(t.size), t] + 1e-12).mean()
        self.probs, self.targets = probs, t
        return logits, float(loss)

    def backward(self):
        """Backprop the mean cross-entropy loss from the last forward()."""
        B, T = self.idx.shape
        dlogits = self.probs.copy()
        dlogits[np.arange(self.targets.size), self.targets] -= 1.0
        dlogits = (dlogits / self.targets.size).reshape(B, T, -1)
        dx = self.ln_f.backward(self.lm_head.backward(dlogits))
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        dx = self.drop.backward(dx)
        np.add.at(self.wte.grad, self.idx, dx)
        self.wpe.grad[:T] += dx.sum(0)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        return list(idx) + list(self.generate_stream(idx, max_new_tokens, temperature, top_k, rng))

    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """Yield sampled token ids one at a time."""
        rng = rng or np.random.default_rng()
        idx = np.asarray(idx, dtype=np.int64).reshape(1, -1)
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx[:, -self.cfg.block_size:])
            logits = logits[0, -1].astype(np.float64) / max(temperature, 1e-8)
            if top_k is not None:
                kth = np.sort(logits)[-min(top_k, logits.size)]
                logits = np.where(logits < kth, -np.inf, logits)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = int(rng.choice(p.size, p=p))
            idx = np.concatenate([idx, [[nxt]]], axis=1)
            yield nxt

    # ------------------------------------------------------------------ checkpointing
    def save(self, path, **extra):
        arrays = {k: p.data for k, p in self.parameters().items()}
        np.savez(path, __config__=np.array(str(asdict(self.cfg))), **arrays,
                 **{f"__extra__{k}": np.array(v) for k, v in extra.items()})

    @classmethod
    def load(cls, path):
        import ast
        f = np.load(path)
        cfg = GPTConfig(**ast.literal_eval(str(f["__config__"])))
        model = cls(cfg)
        for k, p in model.parameters().items():
            p.data = f[k].astype(cfg.dtype)
            p.grad = np.zeros_like(p.data)
        extra = {k[len("__extra__"):]: f[k].item() for k in f.files if k.startswith("__extra__")}
        return model, extra


# ----------------------------------------------------------------------------- optimizer

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * p.grad
            v *= self.b2
            v += (1 - self.b2) * p.grad**2
            update = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if p.data.ndim >= 2:  # decay matrices/embeddings only, not biases/LayerNorm
                update += self.wd * p.data
            p.data -= lr * update


def clip_grad_norm(params, max_norm):
    norm = math.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in params))
    if norm > max_norm:
        for p in params:
            p.grad *= max_norm / (norm + 1e-6)
    return norm
