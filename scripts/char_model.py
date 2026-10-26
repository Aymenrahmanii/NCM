"""Phase 1: a tiny character-level autoregressive transformer.

Trains on data/tinyshakespeare.txt and samples one character at a time.
This is how normal LLMs generate; Phase 2 contrasts it with NCM.

    bash scripts/download_data.sh
    python scripts/char_model.py
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data/tinyshakespeare.txt")
OUT = Path("results/phase1_char_model.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLOCK = 128      # context length in characters
BATCH = 64
DIM = 192
HEADS = 6
LAYERS = 4
STEPS = 3000
LR = 1e-3
EVAL_EVERY = 300


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class CharTransformer(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, DIM)
        self.pos = nn.Embedding(BLOCK, DIM)
        self.blocks = nn.ModuleList(Block(DIM, HEADS) for _ in range(LAYERS))
        self.ln = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        # causal mask: True = not allowed to look (position i can't see j > i)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), 1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))  # (B, T, vocab)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n: int, temperature: float = 0.8) -> torch.Tensor:
        for _ in range(n):  # one forward pass per new character
            logits = self(idx[:, -BLOCK:])[:, -1] / temperature
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def main() -> None:
    torch.manual_seed(0)
    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    def batch(src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(src) - BLOCK - 1, (BATCH,))
        x = torch.stack([src[i : i + BLOCK] for i in ix])
        y = torch.stack([src[i + 1 : i + BLOCK + 1] for i in ix])  # shifted by one
        return x.to(DEVICE), y.to(DEVICE)

    @torch.no_grad()
    def eval_loss() -> tuple[float, float]:
        model.eval()
        out = []
        for src in (train, val):
            losses = []
            for _ in range(20):
                x, y = batch(src)
                losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
            out.append(sum(losses) / len(losses))
        model.train()
        return out[0], out[1]

    model = CharTransformer(len(chars)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab {len(chars)}, params {n_params / 1e6:.2f}M, device {DEVICE}")

    start = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)  # index 0 = "\n"
    log = [f"params: {n_params / 1e6:.2f}M", "", "== sample before training ==",
           decode(model.generate(start, 300)[0].tolist()), "", "== training =="]

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            tr, va = eval_loss()
            line = f"step {step:5d}  train {tr:.3f}  val {va:.3f}"
            print(line)
            log.append(line)
        x, y = batch(train)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())  # forward + loss
        opt.zero_grad()
        loss.backward()                                              # gradients
        opt.step()                                                   # update weights

    model.eval()
    sample = decode(model.generate(start, 500)[0].tolist())
    print(sample)
    log += ["", "== sample after training ==", sample]
    OUT.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
