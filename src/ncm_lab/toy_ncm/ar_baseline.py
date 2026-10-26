"""Autoregressive baseline on the same BPE tokens: one forward pass per token, KV cache on.

    python -m ncm_lab.toy_ncm.ar_baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ART_DIR, BATCH, DIM, EOS_ID, HEADS, PAD_ID, SEQ_LEN, VOCAB_SIZE, device
from .data import load_tokens
from .modules import Transformer, cosine_lr, count_params, sample_logits


class ARModel(nn.Module):
    def __init__(self, layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos = nn.Embedding(SEQ_LEN, DIM)
        self.transformer = Transformer(DIM, HEADS, layers)
        self.head = nn.Linear(DIM, VOCAB_SIZE)
        causal = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool))
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.embed(ids) + self.pos(torch.arange(t, device=ids.device))
        return self.head(self.transformer(x, self.causal[:t, :t]))

    @torch.no_grad()
    def generate(self, batch: int, temperature: float = 1.0, top_k: int | None = None,
                 use_cache: bool = True) -> torch.Tensor:
        dev = self.head.weight.device
        seq = torch.full((batch, 1), EOS_ID, device=dev)  # start token
        caches = [{} for _ in self.transformer.blocks] if use_cache else None
        out = []
        for t in range(SEQ_LEN):
            if use_cache:  # feed only the newest token; old keys/values are cached
                x = self.embed(seq[:, -1:]) + self.pos.weight[t]
                logits = self.head(self.transformer(x, None, caches))[:, -1]
            else:          # recompute the whole prefix every step
                logits = self(seq)[:, -1]
            nxt = sample_logits(logits, temperature, top_k)[:, None]
            out.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
        return torch.cat(out, dim=1)


def load_ar(dev: str) -> ARModel:
    model = ARModel().to(dev)
    model.load_state_dict(torch.load(ART_DIR / "ar.pt", map_location=dev))
    return model.eval()


def train(steps: int = 12000, lr: float = 5e-4) -> None:
    torch.manual_seed(0)
    dev = device()
    train_data, test_data = load_tokens()
    model = ARModel().to(dev)
    print(f"AR params {count_params(model) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def loss_on(ids: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([torch.full_like(ids[:, :1], EOS_ID), ids[:, :-1]], dim=1)
        logits = model(inp)
        return F.cross_entropy(logits.flatten(0, 1).float(), ids.flatten(), ignore_index=PAD_ID)

    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        ids = train_data[torch.randint(len(train_data), (BATCH,))].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss = loss_on(ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=dev == "cuda"):
                val = loss_on(test_data[:512].to(dev)).item()
            model.train()
            print(f"step {step:5d} loss {loss.item():.3f} val {val:.3f}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ART_DIR / "ar.pt")


if __name__ == "__main__":
    train()
