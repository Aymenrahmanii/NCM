"""Phase 12: adapt the XSum NCM and bart-base-xsum to SAMSum with four arms.

    NCM_ROOT=third_party/ncm-pr5-pr4 PYTHONPATH=src python -m ncm_lab.finetune.train \
        --model ncm --arm lora8_qv --seed 0

Arms: zero (no training), lora8_qv (rank 8 on query and value), lora16_all (rank 16 on every
attention and MLP projection), full (every generator weight; the NCM's VQ tokenizer stays
frozen). 3 epochs, pre-registered, no early stopping; validation loss is logged per epoch.
Each finished run appends one row to results/phase12_runs.csv and writes its 819 test
outputs, so the chain can be stopped and restarted between runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ncm_lab.limits.faithfulness import check

from . import lora
from .ncm_loss import ncm_loss, target_rows

RAW = Path("data/samsum/raw")
RESULTS = Path("results")
RUNS_CSV = RESULTS / "phase12_runs.csv"
OUT_DIR = RESULTS / "phase12_outputs"
EPOCHS, BATCH = 3, 16
LR = {"lora8_qv": 3e-4, "lora16_all": 3e-4, "full": 3e-5}
ARMS = ("zero", "lora8_qv", "lora16_all", "full")


def load(split: str) -> list[dict]:
    with open(RAW / f"{split}.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["dialogue"].strip() and r["summary"].strip()]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NCMTask:
    """The released XSum NCM: dialogue -> 512-token prompt, summary -> 48-token canvas."""

    micro = 4  # the prompt is always 512 positions; micro-batch 8 reserves 5 GB in full FT

    def __init__(self, dev: str):
        from ncm_lab.models.ncm import NCMSummarizer

        self.m = NCMSummarizer(device=dev, precision="bf16", kv_cache=False)
        self.net = self.m.net
        # vqvae_local is a plain tuple upstream, so the tokenizer is not a submodule of net
        self.vqvae = self.m.tk.vqvae.requires_grad_(False)
        self.dev = dev

    def batch(self, rows: list[dict]):
        m = self.m
        prefix = m.prepare([r["dialogue"] for r in rows])
        target = target_rows(m.tk.pre_tokenizer, [r["summary"] for r in rows],
                             m.tk.vqvae.context_length, m.eos, m.pad).to(self.dev)
        return prefix, target

    def loss(self, batch) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return ncm_loss(self.net, *batch)

    def trainable(self, arm: str) -> list[torch.nn.Parameter]:
        blk = self.net.transformer.blocks[0].attn
        if arm == "lora8_qv":
            return lora.add_lora(self.net, lora.ncm_qv_selector(blk.n_embd, blk._kv_dim), 8, 16)
        if arm == "lora16_all":
            return lora.add_lora(self.net, lora.ncm_all_selector, 16, 32)
        return [p for p in self.net.parameters() if p.requires_grad]

    def train_mode(self, on: bool) -> None:
        self.net.train(on)
        self.vqvae.eval()

    def generate(self, rows: list[dict]) -> list[str]:
        return self.m.generate([r["dialogue"] for r in rows])  # the released decoding config


class BartTask:
    """morenolq/bart-base-xsum, greedy, 64 new tokens (as in Phase 4)."""

    micro = 8

    def __init__(self, dev: str):
        from ncm_lab.models.baselines import HFSeq2Seq

        self.m = HFSeq2Seq("xsum", device=dev, precision="fp32")
        self.net = self.m.model
        self.dev = dev

    def batch(self, rows: list[dict]):
        enc = self.m.tok([r["dialogue"] for r in rows], return_tensors="pt", padding=True,
                         truncation=True, max_length=512)
        lab = self.m.tok(text_target=[r["summary"] for r in rows], return_tensors="pt",
                         padding=True, truncation=True, max_length=64).input_ids
        lab[lab == self.m.tok.pad_token_id] = -100
        return {**{k: v.to(self.dev) for k, v in enc.items()}, "labels": lab.to(self.dev)}

    def loss(self, batch) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.net(**batch).loss

    def trainable(self, arm: str) -> list[torch.nn.Parameter]:
        if arm == "lora8_qv":
            return lora.add_lora(self.net, lora.bart_qv_selector, 8, 16)
        if arm == "lora16_all":
            return lora.add_lora(self.net, lora.bart_all_selector, 16, 32)
        return [p for p in self.net.parameters() if p.requires_grad]

    def train_mode(self, on: bool) -> None:
        self.net.train(on)

    def generate(self, rows: list[dict]) -> list[str]:
        return self.m.generate([r["dialogue"] for r in rows])


TASKS = {"ncm": NCMTask, "bart": BartTask}


@torch.no_grad()
def val_loss(task, rows: list[dict]) -> float:
    task.train_mode(False)
    m = task.micro
    losses = [task.loss(task.batch(rows[i:i + m])).item() for i in range(0, len(rows), m)]
    return float(np.mean(losses))


def train(task, arm: str, seed: int, train_rows: list[dict], val_rows: list[dict]) -> list:
    params = task.trainable(arm)
    opt = torch.optim.AdamW(params, lr=LR[arm], weight_decay=0.01, betas=(0.9, 0.98))
    steps = EPOCHS * (len(train_rows) // BATCH)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, s / steps))))
    log, step = [], 0
    for epoch in range(EPOCHS):
        order = np.random.default_rng(seed * 1000 + epoch).permutation(len(train_rows))
        task.train_mode(True)
        for b in range(len(train_rows) // BATCH):
            idx = order[b * BATCH:(b + 1) * BATCH]
            for i in range(0, BATCH, task.micro):
                loss = task.loss(task.batch([train_rows[j] for j in idx[i:i + task.micro]]))
                (loss * task.micro / BATCH).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f}", flush=True)
        log.append(round(val_loss(task, val_rows), 4))
        print(f"  epoch {epoch + 1}: val loss {log[-1]}", flush=True)
    task.train_mode(False)
    return log


def evaluate(hyps: list[str], rows: list[dict]) -> dict:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = [sc.score(r["summary"], h) for h, r in zip(hyps, rows)]
    faith = [check(h, r["dialogue"], lenient=True) for h, r in zip(hyps, rows)]
    n = len(rows)
    return {**{k: round(100 * statistics.fmean(x[k].fmeasure for x in s), 2)
               for k in ("rouge1", "rouge2", "rougeL")},
            "invented_numbers": round(sum(bool(f["invented_numbers"]) for f in faith) / n, 4),
            "invented_names": round(sum(bool(f["invented_names"]) for f in faith) / n, 4),
            "repeats": round(sum(f["repeats"] for f in faith) / n, 4),
            "words": round(statistics.fmean(len(h.split()) for h in hyps), 1)}


def done(model: str, arm: str, seed: int) -> bool:
    if not RUNS_CSV.exists():
        return False
    with open(RUNS_CSV, newline="") as f:
        return any(r["model"] == model and r["arm"] == arm and r["seed"] == str(seed)
                   for r in csv.DictReader(f))


def run(model: str, arm: str, seed: int, dev: str = "cuda") -> None:
    if done(model, arm, seed):
        print(f"{model} {arm} seed {seed}: already done", flush=True)
        return
    seed_all(seed)
    task = TASKS[model](dev)
    train_rows, val_rows, test_rows = load("train"), load("validation"), load("test")
    t0 = time.time()
    log = [] if arm == "zero" else train(task, arm, seed, train_rows, val_rows)
    minutes = round((time.time() - t0) / 60, 1)
    n_train = 0 if arm == "zero" else sum(
        p.numel() for p in task.net.parameters() if p.requires_grad)
    with torch.no_grad():
        hyps = task.generate(test_rows)
    row = {"model": model, "arm": arm, "seed": seed, "trainable_m": round(n_train / 1e6, 2),
           "train_min": minutes, "val_loss": json.dumps(log), **evaluate(hyps, test_rows)}
    print(row, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{model}_{arm}_s{seed}.jsonl", "w", encoding="utf-8") as f:
        for h, r in zip(hyps, test_rows):
            f.write(json.dumps({"id": r["id"], "hyp": h}) + "\n")
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def rescore() -> None:
    """Recompute every run's metrics from its saved outputs (after a metric change), and
    add the human reference summaries as a calibration row."""
    test_rows = load("test")
    with open(RUNS_CSV, newline="") as f:
        runs = [r for r in csv.DictReader(f) if r["model"] != "reference"]
    out = []
    for r in runs:
        path = OUT_DIR / f"{r['model']}_{r['arm']}_s{r['seed']}.jsonl"
        with open(path, encoding="utf-8") as f:
            hyps = [json.loads(line)["hyp"] for line in f]
        out.append({**r, **evaluate(hyps, test_rows)})
    ref = evaluate([r["summary"] for r in test_rows], test_rows)
    out.append({**{k: "" for k in out[0]}, "model": "reference", "arm": "human", **ref})
    with open(RUNS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(TASKS))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rescore", action="store_true", help="recompute metrics, no training")
    a = p.parse_args(argv)
    if a.rescore:
        rescore()
    elif a.model and a.arm:
        run(a.model, a.arm, a.seed)
    else:
        p.error("--model and --arm are required unless --rescore")


if __name__ == "__main__":
    main()
