"""Phase 12 gate: can the XSum checkpoint's frozen VQ tokenizer rebuild SAMSum summaries?

    NCM_ROOT=third_party/ncm-pr5-pr4 PYTHONPATH=src python scripts/phase12_gate.py

Each summary is GPT-2 tokenized, closed with <eos> and padded to the tokenizer's 48
positions (longer ones are cut, and counted). Reports token accuracy over the real positions
(summary tokens and <eos>) and the share of summaries rebuilt exactly. Gate: >= 95% tokens.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from ncm_lab.models.ncm import NCMSummarizer

RAW = Path("data/samsum/raw")
OUT = Path("results/phase12_gate.json")


def summaries(split: str) -> list[str]:
    with open(RAW / f"{split}.csv", encoding="utf-8") as f:
        return [r["summary"].strip() for r in csv.DictReader(f)]


@torch.no_grad()
def gate(model: NCMSummarizer, texts: list[str], batch: int = 64) -> dict:
    tk, vq = model.tk, model.tk.vqvae
    n_ctx = vq.context_length
    dev = next(vq.parameters()).device
    rows, cut = [], 0
    for t in texts:
        ids = tk.pre_tokenizer.encode(t)[: n_ctx - 1] + [model.eos]
        cut += len(tk.pre_tokenizer.encode(t)) > n_ctx - 1
        rows.append(ids + [model.pad] * (n_ctx - len(ids)))
    x = torch.tensor(rows, device=dev)
    real = torch.zeros_like(x, dtype=torch.bool)
    for i, t in enumerate(rows):
        real[i, : t.index(model.eos) + 1] = True
    # vq.reconstruct is broken upstream (vq returns more values now); encode -> decode instead
    pred = torch.cat([vq.decode(vq.encode(c)).argmax(-1) for c in x.split(batch)])
    hit = (pred == x) & real
    return {"n": len(texts), "truncated": cut,
            "token_acc": round(hit.sum().item() / real.sum().item(), 4),
            "exact": round(float((hit.sum(1) == real.sum(1)).float().mean()), 4),
            "mean_len": round(real.sum().item() / len(texts), 1)}


def main() -> None:
    model = NCMSummarizer(device="cuda" if torch.cuda.is_available() else "cpu")
    out = {split: gate(model, summaries(split)) for split in ("validation", "test")}
    print(json.dumps(out, indent=1))
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
