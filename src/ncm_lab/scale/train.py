"""Resumable Phase 10 training for the tokenizer, NCM generators and the AR ladder.

    python -m ncm_lab.scale.train tokenizer
    python -m ncm_lab.scale.train ar --size ar7m --seed 0
    python -m ncm_lab.scale.train gen --size gen25m --seed 0

Every run lives in artifacts/scale/<name>/: `ckpt.pt` (saved atomically every
`--save-every` steps), `log.csv`, and `model.pt` once finished. Rerunning the same command
resumes. Batches are drawn from a generator seeded by (seed, step), so a resumed run sees
exactly the batches an uninterrupted one would. Creating artifacts/scale/STOP makes every
run save and exit at its next step (to free the GPU).

Constant across models (plan): AdamW, cosine schedule with 300 warmup steps, bf16 autocast,
batch 64 x 256 tokens, gradient clip 1.0, same lr.
"""

from __future__ import annotations

import argparse
import dataclasses
import csv
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_lab.toy_ncm.config import PAD_ID
from ncm_lab.toy_ncm.modules import cosine_lr, count_params

from .data import batch_from, load_split
from .models import (MIXERS, ARConfig, ARModel, Generator, GeneratorConfig, Tokenizer,
                     TokenizerConfig, config_dict)

ART = Path("artifacts/scale")
STOP_FILE = ART / "STOP"
BATCH = 64
LR = 5e-4
TOKENS_PER_STEP = BATCH * 256

TOKENIZER = TokenizerConfig(dim=288, heads=6, enc_layers=4, dec_layers=4)  # 10.7M
# Validity check: the official TinyStories NCM's level schedule (1, 8, 16, ..., 108, 128 at
# length 128), doubled for 256 tokens, and its 4096 codes per level. The finest levels all
# work near full length, so the last level only corrects a small residual.
TOKENIZERS = {
    "tokenizer": TOKENIZER,
    "tokv2": dataclasses.replace(TOKENIZER, n_codes=4096, levels=(
        1, 16, 32, 48, 64, 80, 104, 128, 152, 184, 216, 256)),
}
GENERATORS = {"gen12m": GeneratorConfig(384, 6, 6), "gen25m": GeneratorConfig(512, 8, 7)}
AR_LADDER = {
    "ar2m": ARConfig(dim=128, heads=4, layers=4),     # 1.9M
    "ar3m": ARConfig(dim=192, heads=6, layers=4),     # 3.4M
    "ar7m": ARConfig(dim=256, heads=4, layers=6),     # 6.9M
    "ar25m": ARConfig(dim=512, heads=8, layers=7),    # 26.4M
    "ar35m": ARConfig(dim=512, heads=8, layers=10),   # 35.9M = tokenizer + gen25m
}
DEFAULT_TOKENS = {"tokenizer": 100e6, "gen": 200e6, "ar": 200e6}
MAX_CORRUPT = 0.25
VAL_ROWS = 1024
EVAL_CHUNK = 64
# Speed settings, used on CUDA only. None of them changes the maths (compile and the fused
# optimizer only reorder floating-point work), and checkpoints stay interchangeable:
# Module.compile() compiles in place, so state_dict keys do not change.
SPEED = {"in_memory": True, "fused_adam": True, "compile": True}


def amp(dev: str):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")


def batch_indices(n: int, seed: int, step: int, size: int = BATCH) -> np.ndarray:
    return np.random.default_rng([seed, step]).integers(0, n, size)


def corrupt(codes: torch.Tensor, rate: float, n_codes: int) -> torch.Tensor:
    noise = torch.rand(codes.shape, device=codes.device) < rate
    return torch.where(noise, torch.randint_like(codes, n_codes), codes)


# ---- per-kind losses and evals --------------------------------------------------------

def tokenizer_loss(tok: Tokenizer, ids: torch.Tensor) -> torch.Tensor:
    """Reconstruction CE with nested dropout (half the batch decodes from a level prefix)."""
    n_levels, b = len(tok.cfg.levels), len(ids)
    _, f_hats, vq_loss = tok.quantize(tok.encode(ids))
    keep = torch.randint(1, n_levels + 1, (b,), device=ids.device)
    keep = torch.where(torch.rand(b, device=ids.device) < 0.5, n_levels, keep)
    f_sel = torch.stack(f_hats)[keep - 1, torch.arange(b, device=ids.device)]
    ce = F.cross_entropy(tok.decode(f_sel).flatten(0, 1).float(), ids.flatten(),
                         ignore_index=PAD_ID)
    return ce + vq_loss


@torch.no_grad()
def tokenizer_eval(tok: Tokenizer, val: torch.Tensor) -> dict:
    hits = torch.zeros(len(tok.cfg.levels))
    for i in range(0, len(val), EVAL_CHUNK):  # full-vocab logits: keep chunks small
        ids = val[i:i + EVAL_CHUNK]
        keep = ids != PAD_ID
        _, f_hats, _ = tok.quantize(tok.encode(ids))
        for k, f in enumerate(f_hats):
            hits[k] += ((tok.decode(f).argmax(-1) == ids) & keep).sum().item()
    total = (val != PAD_ID).sum().item()
    return {f"acc_l{k + 1}": round(h.item() / total, 4) for k, h in enumerate(hits)}


def generator_losses(gen: Generator, tok: Tokenizer, ids: torch.Tensor,
                     rate: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with torch.no_grad():
        codes, _, _ = tok.quantize(tok.encode(ids))
        noisy = [corrupt(c, rate, tok.cfg.n_codes) for c in codes]
        after = tok.codes_to_fhats(noisy)
        canvases = [torch.zeros_like(after[0])] + after[:-1]
    logits = gen(canvases)
    losses = [F.cross_entropy(lg.flatten(0, 1).float(), c.flatten())
              for lg, c in zip(logits, codes)]
    return losses, [(lg.argmax(-1) == c).float().mean() for lg, c in zip(logits, codes)]


@torch.no_grad()
def generator_eval(gen: Generator, tok: Tokenizer, val: torch.Tensor) -> dict:
    out = {}
    for i in range(0, len(val), EVAL_CHUNK):
        losses, accs = generator_losses(gen, tok, val[i:i + EVAL_CHUNK], 0.0)
        w = len(val[i:i + EVAL_CHUNK]) / len(val)
        for k, (lo, ac) in enumerate(zip(losses, accs)):
            out[f"nll_l{k + 1}"] = out.get(f"nll_l{k + 1}", 0.0) + lo.item() * w
            out[f"acc_l{k + 1}"] = out.get(f"acc_l{k + 1}", 0.0) + ac.item() * w
    return {k: round(v, 4) for k, v in out.items()}


@torch.no_grad()
def ar_eval(model: ARModel, val: torch.Tensor) -> dict:
    nll = sum(model.loss(val[i:i + EVAL_CHUNK], PAD_ID).item() * len(val[i:i + EVAL_CHUNK])
              for i in range(0, len(val), EVAL_CHUNK)) / len(val)
    return {"val_nll": round(nll, 4)}


# ---- the shared loop ------------------------------------------------------------------

def run(name: str, model: nn.Module, loss_fn: Callable[[torch.Tensor, int], torch.Tensor],
        eval_fn: Callable[[], dict], meta: dict, tokens: float, seed: int,
        micro_batch: int = BATCH, save_every: int = 250, eval_every: int = 1000,
        on_step: Callable[[int, int], None] | None = None, max_minutes: float | None = None,
        dev: str = "cuda") -> bool:
    """Train `model` for `tokens` tokens. Returns True when finished, False if stopped."""
    out = ART / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        print(f"{name}: already finished", flush=True)
        return True
    total = int(tokens // TOKENS_PER_STEP)
    cuda = dev == "cuda"
    train = load_split("train", in_memory=cuda and SPEED["in_memory"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                            weight_decay=0.01, betas=(0.9, 0.95),
                            fused=cuda and SPEED["fused_adam"])
    if cuda and SPEED["compile"]:
        model.compile()
    step = 0
    if (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"{name}: resumed at step {step}/{total}", flush=True)
    else:
        print(f"{name}: {count_params(model) / 1e6:.2f}M params, {total} steps "
              f"({tokens / 1e6:.0f}M tokens)", flush=True)

    def save(path: Path, payload) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def checkpoint() -> None:
        save(out / "ckpt.pt", {"model": model.state_dict(), "opt": opt.state_dict(),
                               "step": step, "meta": meta})

    log = open(out / "log.csv", "a", newline="")
    start, deadline = time.time(), (time.time() + 60 * max_minutes if max_minutes else None)
    model.train()
    while step < total:
        if STOP_FILE.exists() or (deadline and time.time() > deadline):
            checkpoint()
            print(f"{name}: stopped at step {step}/{total}", flush=True)
            log.close()
            return False
        torch.manual_seed(seed * 100_003 + step)  # level dropout, corruption: resumable
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, total, LR)
        idx = batch_indices(len(train), seed, step)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for j in range(0, BATCH, micro_batch):
            ids = batch_from(train, idx[j:j + micro_batch], dev)
            with amp(dev):
                loss = loss_fn(ids, step) * (len(ids) / BATCH)
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if on_step:
            on_step(step, total)
        step += 1

        if step % eval_every == 0 or step == total:
            model.eval()
            with amp(dev):
                metrics = eval_fn()
            model.train()
            if dev == "cuda":  # give back the eval's peak memory, or it stays reserved
                torch.cuda.empty_cache()
            row = {"step": step, "tokens_m": round(step * TOKENS_PER_STEP / 1e6, 1),
                   "train_loss": round(loss_sum, 4), "lr": cosine_lr(step - 1, total, LR),
                   "min_elapsed": round((time.time() - start) / 60, 1), **metrics}
            w = csv.DictWriter(log, list(row))
            if log.tell() == 0:
                w.writeheader()
            w.writerow(row)
            log.flush()
            print(f"{name} {row}", flush=True)
        if step % save_every == 0:
            checkpoint()
    save(out / "model.pt", {"model": model.state_dict(), "meta": meta})
    (out / "ckpt.pt").unlink(missing_ok=True)
    log.close()
    print(f"{name}: finished", flush=True)
    return True


def load_val(dev: str) -> torch.Tensor:
    val = load_split("val")
    return torch.from_numpy(val[:VAL_ROWS].astype(np.int64)).to(dev)


def load_tokenizer(dev: str, name: str = "tokenizer") -> Tokenizer:
    ck = torch.load(ART / name / "model.pt", map_location=dev, weights_only=False)
    tok = Tokenizer(TokenizerConfig(**ck["meta"]["config"])).to(dev)
    tok.load_state_dict(ck["model"])
    return tok.eval().requires_grad_(False)


def train_tokenizer(tokens: float, seed: int, dev: str, name: str = "tokenizer",
                    **kw) -> bool:
    torch.manual_seed(seed)
    cfg = TOKENIZERS[name]
    tok = Tokenizer(cfg).to(dev)
    val = load_val(dev)

    def on_step(step: int, total: int) -> None:
        if step % 200 == 0 and step < 0.8 * total:
            for vq in tok.quantizers:
                vq.restart_dead()

    return run(name, tok, lambda ids, s: tokenizer_loss(tok, ids),
               lambda: tokenizer_eval(tok, val), {"kind": "tokenizer",
                                                  "config": config_dict(cfg)},
               tokens, seed, on_step=on_step, dev=dev, **kw)


def generator_run_name(size: str, mixer: str, tokens: float, seed: int,
                       tok: str = "tokenizer", tied: bool = False) -> str:
    """Phase 10 names stay as they were; Phase 11 variants and budgets get their own."""
    default = mixer == "attn" and tokens == DEFAULT_TOKENS["gen"]
    if default and tok == "tokenizer" and not tied:
        return f"{size}_s{seed}"
    extra = (f"-{tok}" if tok != "tokenizer" else "") + ("-tied" if tied else "")
    return f"{size}-{mixer}-{tokens / 1e6:g}m{extra}_s{seed}"


def train_generator(size: str, tokens: float, seed: int, dev: str, mixer: str = "attn",
                    tok_name: str = "tokenizer", tied: bool = False, **kw) -> bool:
    torch.manual_seed(seed)
    tok = load_tokenizer(dev, tok_name)
    if dev == "cuda" and SPEED["compile"]:
        tok.encoder.compile()  # frozen, but runs on every batch to produce the targets
    cfg = dataclasses.replace(GENERATORS[size], mixer=mixer, tied_heads=tied)
    codebooks = [vq.codebook.detach() for vq in tok.quantizers] if tied else None
    gen = Generator(cfg, tok.cfg, codebooks).to(dev)
    val = load_val(dev)

    def loss_fn(ids: torch.Tensor, step: int) -> torch.Tensor:
        losses, _ = generator_losses(gen, tok, ids, torch.rand(1).item() * MAX_CORRUPT)
        return torch.stack(losses).mean()

    meta = {"kind": "gen", "config": config_dict(cfg), "seed": seed, "tokenizer": tok_name}
    return run(generator_run_name(size, mixer, tokens, seed, tok_name, tied), gen, loss_fn,
               lambda: generator_eval(gen, tok, val), meta, tokens, seed, dev=dev, **kw)


def train_ar(size: str, tokens: float, seed: int, dev: str, **kw) -> bool:
    torch.manual_seed(seed)
    model = ARModel(AR_LADDER[size]).to(dev)
    val = load_val(dev)
    meta = {"kind": "ar", "config": config_dict(AR_LADDER[size]), "seed": seed}
    return run(f"{size}_s{seed}", model, lambda ids, s: model.loss(ids, PAD_ID),
               lambda: ar_eval(model, val), meta, tokens, seed, dev=dev, **kw)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["tokenizer", "gen", "ar"])
    p.add_argument("--size", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokens", type=float, default=None)
    p.add_argument("--micro-batch", type=int, default=BATCH)
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--mixer", choices=MIXERS, default="attn", help="generator only (Phase 11)")
    p.add_argument("--tok", choices=list(TOKENIZERS), default="tokenizer",
                   help="tokenizer to train, or the one a generator uses")
    p.add_argument("--tied-heads", action="store_true", help="generator: codebook-tied heads")
    a = p.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = a.tokens or DEFAULT_TOKENS[a.kind]
    kw = {"micro_batch": a.micro_batch, "max_minutes": a.max_minutes}
    if a.kind == "tokenizer":
        train_tokenizer(tokens, a.seed, dev, name=a.tok, **kw)
    elif a.kind == "gen":
        train_generator(a.size or "gen25m", tokens, a.seed, dev, mixer=a.mixer,
                        tok_name=a.tok, tied=a.tied_heads, **kw)
    else:
        train_ar(a.size or "ar7m", tokens, a.seed, dev, **kw)


if __name__ == "__main__":
    main()
