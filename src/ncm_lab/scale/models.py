"""Configurable versions of the toy NCM tokenizer, generator and AR baseline.

Same design as `ncm_lab.toy_ncm` (see its docstrings), with every shape a parameter, so
Phase 10 can train 256-token models with 9 levels and an AR ladder of several sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_lab.toy_ncm.config import EOS_ID, VOCAB_SIZE
from ncm_lab.toy_ncm.modules import Attention, Transformer, sample_logits
from ncm_lab.toy_ncm.tokenizer import VectorQuantizer, down, up


@dataclass(frozen=True)
class TokenizerConfig:
    seq_len: int = 256
    levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    n_codes: int = 512
    code_dim: int = 32
    dim: int = 320
    heads: int = 5
    enc_layers: int = 4
    dec_layers: int = 4


MIXERS = ("attn", "nocross", "conv", "linear")


@dataclass(frozen=True)
class GeneratorConfig:
    dim: int = 384
    heads: int = 6
    layers: int = 6
    # Phase 11 token mixers (see GenMixer): attn is the Phase 10 generator.
    mixer: str = "attn"
    conv_from: int = 64  # conv mixer: levels at least this long use a depthwise convolution
    conv_kernel: int = 7


@dataclass(frozen=True)
class ARConfig:
    seq_len: int = 256
    dim: int = 256
    heads: int = 4
    layers: int = 6


def config_dict(cfg) -> dict:
    return asdict(cfg)


class Tokenizer(nn.Module):
    """Multiscale residual VQ: text -> codes per level -> canvas -> text."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.embed = nn.Embedding(VOCAB_SIZE, d)
        self.enc_pos = nn.Parameter(torch.randn(1, cfg.seq_len, d) * 0.02)
        self.encoder = Transformer(d, cfg.heads, cfg.enc_layers)
        self.to_code = nn.Linear(d, cfg.code_dim)
        self.pre_quant_norm = nn.LayerNorm(cfg.code_dim, elementwise_affine=False)
        self.quantizers = nn.ModuleList(VectorQuantizer(cfg.n_codes, cfg.code_dim)
                                        for _ in cfg.levels)
        self.from_code = nn.Linear(cfg.code_dim, d)
        self.dec_pos = nn.Parameter(torch.randn(1, cfg.seq_len, d) * 0.02)
        self.decoder = Transformer(d, cfg.heads, cfg.dec_layers)
        self.head = nn.Linear(d, VOCAB_SIZE)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pre_quant_norm(self.to_code(self.encoder(self.embed(ids) + self.enc_pos)))

    def quantize(self, z: torch.Tensor):
        """Codes per level, canvas after each level, VQ loss (see toy_ncm.tokenizer)."""
        n = self.cfg.seq_len
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            f_hat = torch.zeros_like(z)
            codes, f_hats, loss = [], [], 0.0
            for length, vq in zip(self.cfg.levels, self.quantizers):
                r = down(z - f_hat.detach(), length)
                q, idx, level_loss = vq(r)
                q = r + (q - r).detach()
                f_hat = f_hat + up(q, n)
                codes.append(idx)
                f_hats.append(f_hat)
                loss = loss + level_loss
        return codes, f_hats, loss / len(self.cfg.levels)

    def decode(self, f_hat: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.from_code(f_hat) + self.dec_pos))

    def lookup_up(self, k: int, idx: torch.Tensor) -> torch.Tensor:
        """Level k's codes as a full-length canvas increment."""
        return up(self.quantizers[k].lookup(idx).float(), self.cfg.seq_len)

    def codes_to_fhats(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        f_hat, out = 0.0, []
        for k, idx in enumerate(codes):
            f_hat = f_hat + self.lookup_up(k, idx)
            out.append(f_hat)
        return out


class Generator(nn.Module):
    """Predicts every code of level k from the canvas of levels 1..k-1 (block-causal)."""

    def __init__(self, cfg: GeneratorConfig, tok: TokenizerConfig):
        super().__init__()
        self.cfg, self.tok = cfg, tok
        levels, n = tok.levels, tok.seq_len
        self.in_proj = nn.Linear(tok.code_dim, cfg.dim)
        self.level_emb = nn.Embedding(len(levels), cfg.dim)
        self.pos_emb = nn.Embedding(n, cfg.dim)
        assert cfg.mixer in MIXERS, cfg.mixer
        self.transformer = (Transformer(cfg.dim, cfg.heads, cfg.layers) if cfg.mixer == "attn"
                            else GenTransformer(cfg, tok))
        self.heads = nn.ModuleList(nn.Linear(cfg.dim, tok.n_codes) for _ in levels)
        level_ids = torch.cat([torch.full((m,), k) for k, m in enumerate(levels)])
        pos_ids = torch.cat([((torch.arange(m) + 0.5) * n / m).long() for m in levels])
        self.register_buffer("level_ids", level_ids, persistent=False)
        self.register_buffer("pos_ids", pos_ids, persistent=False)
        self.register_buffer("mask", level_ids[:, None] >= level_ids[None, :], persistent=False)
        self.starts = [0]
        for m in levels:
            self.starts.append(self.starts[-1] + m)

    def forward(self, canvases: list[torch.Tensor]) -> list[torch.Tensor]:
        """canvases[k] = canvas before level k, (B, seq_len, code_dim). Logits per level."""
        k = len(canvases)
        t = self.starts[k]
        x = torch.cat([down(c, m) for c, m in zip(canvases, self.tok.levels)], dim=1)
        x = self.in_proj(x) + self.level_emb(self.level_ids[:t]) + self.pos_emb(self.pos_ids[:t])
        if self.cfg.mixer == "attn":
            h = self.transformer(x, self.mask[:t, :t])
        else:
            h = self.transformer(x, k)
        return [self.heads[i](h[:, self.starts[i]: self.starts[i + 1]]) for i in range(k)]


class GenMixer(nn.Module):
    """Token mixing for the Phase 11 generator variants, on the first k levels.

    Every position's input already holds the canvas of all coarser levels, so mixing across
    levels may be redundant. nocross: attention inside each level only. conv: depthwise
    convolution inside each level of conv_from positions or more, block-causal attention
    among the shorter levels. linear: block-causal linear attention (elu + 1 feature map).
    """

    def __init__(self, cfg: GeneratorConfig, tok: TokenizerConfig):
        super().__init__()
        self.cfg, self.levels = cfg, tok.levels
        self.heads = cfg.heads
        self.starts = [0]
        for m in tok.levels:
            self.starts.append(self.starts[-1] + m)
        level_ids = torch.cat([torch.full((m,), k) for k, m in enumerate(tok.levels)])
        self.register_buffer("level_ids", level_ids, persistent=False)
        if cfg.mixer in ("nocross", "conv"):
            self.attn = Attention(cfg.dim, cfg.heads)
            same = level_ids[:, None] == level_ids[None, :]
            causal = level_ids[:, None] >= level_ids[None, :]
            self.register_buffer("mask", same if cfg.mixer == "nocross" else causal,
                                 persistent=False)
        if cfg.mixer == "conv":
            self.n_coarse = sum(m < cfg.conv_from for m in tok.levels)
            self.conv = nn.Conv1d(cfg.dim, cfg.dim, cfg.conv_kernel, padding=cfg.conv_kernel // 2,
                                  groups=cfg.dim)
            self.conv_proj = nn.Linear(cfg.dim, cfg.dim)
        if cfg.mixer == "linear":
            self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
            self.proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        t = self.starts[k]
        if self.cfg.mixer == "nocross":
            return self.attn(x, self.mask[:t, :t])
        if self.cfg.mixer == "conv":
            c = self.starts[min(k, self.n_coarse)]
            parts = [self.attn(x[:, :c], self.mask[:c, :c])] if c else []
            for lv in range(self.n_coarse, k):  # each fine level on its own: no leakage
                seg = x[:, self.starts[lv]: self.starts[lv + 1]].transpose(1, 2)
                parts.append(self.conv_proj(self.conv(seg).transpose(1, 2)))
            return torch.cat(parts, dim=1)
        return self._linear(x, k)

    def _linear(self, x: torch.Tensor, k: int) -> torch.Tensor:
        b, t, c = x.shape
        q, kk, v = self.qkv(x).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, kk = F.elu(q.float()) + 1, F.elu(kk.float()) + 1
        v = v.float()
        ids = self.level_ids[:t]
        one_hot = F.one_hot(ids, k).to(q.dtype)  # (t, k): position -> level
        kv = torch.einsum("bhtd,bhte,tl->bhlde", kk, v, one_hot).cumsum(2)  # levels <= l
        z = torch.einsum("bhtd,tl->bhld", kk, one_hot).cumsum(2)
        outs = []
        for lv in range(k):  # per level, not per position: (t, d, d) would not fit in memory
            ql = q[:, :, self.starts[lv]: self.starts[lv + 1]]
            den = (ql * z[:, :, lv, None]).sum(-1, keepdim=True).clamp(min=1e-6)
            outs.append(ql @ kv[:, :, lv] / den)
        out = torch.cat(outs, dim=2).to(x.dtype)
        return self.proj(out.transpose(1, 2).reshape(b, t, c))


class GenBlock(nn.Module):
    def __init__(self, cfg: GeneratorConfig, tok: TokenizerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.mix = GenMixer(cfg, tok)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
                                 nn.Linear(4 * cfg.dim, cfg.dim))

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        x = x + self.mix(self.ln1(x), k)
        return x + self.mlp(self.ln2(x))


class GenTransformer(nn.Module):
    def __init__(self, cfg: GeneratorConfig, tok: TokenizerConfig):
        super().__init__()
        self.blocks = nn.ModuleList(GenBlock(cfg, tok) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.dim)

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, k)
        return self.ln(x)


@torch.no_grad()
def ncm_sample(gen: Generator, tok: Tokenizer, batch: int,
               temperature: float | list[float] = 1.0, top_k: int | None = None,
               prefix: list[torch.Tensor] | None = None):
    """Level by level, one forward pass each. 0 temperature = greedy. Returns codes, canvases."""
    dev = next(gen.parameters()).device
    levels = tok.cfg.levels
    prefix = prefix or []
    temps = temperature if isinstance(temperature, list) else [temperature] * len(levels)
    canvas = torch.zeros(batch, tok.cfg.seq_len, tok.cfg.code_dim, device=dev)
    inputs, codes, canvases = [], [], []
    for k in range(len(levels)):
        inputs.append(canvas)
        if k < len(prefix):
            idx = prefix[k]
        elif temps[k] == 0.0:
            idx = gen(inputs)[k].argmax(-1)
        else:
            idx = sample_logits(gen(inputs)[k], temps[k], top_k)
        codes.append(idx)
        canvas = canvas + tok.lookup_up(k, idx)
        canvases.append(canvas)
    return codes, canvases


class ARModel(nn.Module):
    """Decoder-only baseline on the same BPE tokens, with a KV cache for generation."""

    def __init__(self, cfg: ARConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.dim)
        self.pos = nn.Embedding(cfg.seq_len, cfg.dim)
        self.transformer = Transformer(cfg.dim, cfg.heads, cfg.layers)
        self.head = nn.Linear(cfg.dim, VOCAB_SIZE)
        causal = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool))
        self.register_buffer("causal", causal, persistent=False)

    use_flash = True  # is_causal instead of an explicit mask: same maths, faster kernel

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.embed(ids) + self.pos(torch.arange(t, device=ids.device))
        if self.use_flash:
            return self.head(self.transformer(x, causal=True))
        return self.head(self.transformer(x, self.causal[:t, :t]))

    def loss(self, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        inp = torch.cat([torch.full_like(ids[:, :1], EOS_ID), ids[:, :-1]], dim=1)
        return F.cross_entropy(self(inp).flatten(0, 1).float(), ids.flatten(),
                               ignore_index=pad_id)

    @torch.no_grad()
    def generate(self, batch: int, temperature: float = 1.0, top_k: int | None = None,
                 length: int | None = None) -> torch.Tensor:
        dev = self.head.weight.device
        length = length or self.cfg.seq_len
        last = torch.full((batch, 1), EOS_ID, device=dev)
        caches = [{} for _ in self.transformer.blocks]
        out = []
        for t in range(length):
            x = self.embed(last) + self.pos.weight[t]
            logits = self.head(self.transformer(x, None, caches))[:, -1]
            last = (logits.argmax(-1) if temperature == 0.0
                    else sample_logits(logits, temperature, top_k))[:, None]
            out.append(last)
        return torch.cat(out, dim=1)
