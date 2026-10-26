"""Small transformer pieces shared by the tokenizer, generator and AR baseline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None, cache: dict | None = None):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        if cache is not None:  # KV cache: keep keys/values of earlier tokens
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # mask: bool (T, S), True = may attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, mask=None, cache=None):
        x = x + self.attn(self.ln1(x), mask, cache)
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None, caches: list[dict] | None = None):
        for i, block in enumerate(self.blocks):
            x = block(x, mask, None if caches is None else caches[i])
        return self.ln(x)


def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None):
    """Sample one index per row of logits (..., V)."""
    logits = logits.float() / max(temperature, 1e-5)
    if top_k is not None:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
    return flat.view(probs.shape[:-1])


def cosine_lr(step: int, total: int, base: float, warmup: int = 300) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
