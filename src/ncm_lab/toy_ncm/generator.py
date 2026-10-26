"""Stage 2: the generator.

Given the canvas from all earlier levels (shrunk to the current level's length), predict
every code of the current level at once. All levels train in one forward pass with a
block-causal mask: positions of level k see levels 1..k, never later ones.

    python -m ncm_lab.toy_ncm.generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ART_DIR, BATCH, CODE_DIM, DIM, HEADS, LEVELS, N_CODES, SEQ_LEN, device
from .data import load_tokens
from .modules import Transformer, cosine_lr, count_params, sample_logits
from .tokenizer import MultiscaleTokenizer, down, load_tokenizer, up


class Generator(nn.Module):
    def __init__(self, layers: int = 6):
        super().__init__()
        self.in_proj = nn.Linear(CODE_DIM, DIM)
        self.level_emb = nn.Embedding(len(LEVELS), DIM)
        self.pos_emb = nn.Embedding(SEQ_LEN, DIM)  # position on the 64-token grid
        self.transformer = Transformer(DIM, HEADS, layers)
        self.heads = nn.ModuleList(nn.Linear(DIM, N_CODES) for _ in LEVELS)

        level_ids = torch.cat([torch.full((n,), k) for k, n in enumerate(LEVELS)])
        pos_ids = torch.cat([((torch.arange(n) + 0.5) * SEQ_LEN / n).long() for n in LEVELS])
        self.register_buffer("level_ids", level_ids, persistent=False)
        self.register_buffer("pos_ids", pos_ids, persistent=False)
        # True = may attend: query level >= key level
        self.register_buffer("mask", level_ids[:, None] >= level_ids[None, :], persistent=False)
        self.starts = [0]
        for n in LEVELS:
            self.starts.append(self.starts[-1] + n)

    def forward(self, canvases: list[torch.Tensor]) -> list[torch.Tensor]:
        """canvases[k] = canvas before level k, (B, 64, CODE_DIM). Returns logits per level."""
        k = len(canvases)
        t = self.starts[k]
        x = torch.cat([down(c, n) for c, n in zip(canvases, LEVELS)], dim=1)
        x = self.in_proj(x) + self.level_emb(self.level_ids[:t]) + self.pos_emb(self.pos_ids[:t])
        h = self.transformer(x, self.mask[:t, :t])
        return [self.heads[i](h[:, self.starts[i] : self.starts[i + 1]]) for i in range(k)]


@torch.no_grad()
def sample(gen: Generator, tok: MultiscaleTokenizer, batch: int, temperature: float = 1.0,
           top_k: int | None = None):
    """Generate level 1, then level 2 from it, ... One forward pass per level.

    Returns the codes per level and the canvas after each level.
    """
    dev = next(gen.parameters()).device
    canvas = torch.zeros(batch, SEQ_LEN, CODE_DIM, device=dev)
    inputs, codes, canvases = [], [], []
    for k in range(len(LEVELS)):
        inputs.append(canvas)
        logits = gen(inputs)[k]
        idx = sample_logits(logits, temperature, top_k)
        codes.append(idx)
        canvas = canvas + up(tok.quantizers[k].lookup(idx))
        canvases.append(canvas)
    return codes, canvases


def corrupt(codes: torch.Tensor, rate: float) -> torch.Tensor:
    """Replace a fraction of codes with random ones (teaches recovery from own mistakes)."""
    noise = torch.rand(codes.shape, device=codes.device) < rate
    return torch.where(noise, torch.randint_like(codes, N_CODES), codes)


def load_generator(dev: str) -> Generator:
    gen = Generator().to(dev)
    gen.load_state_dict(torch.load(ART_DIR / "generator.pt", map_location=dev))
    return gen.eval()


def train(steps: int = 12000, lr: float = 5e-4, max_corrupt: float = 0.25) -> None:
    torch.manual_seed(0)
    dev = device()
    train_data, test_data = load_tokens()
    tok = load_tokenizer(dev)  # frozen
    gen = Generator().to(dev)
    print(f"generator params {count_params(gen) / 1e6:.2f}M")
    opt = torch.optim.AdamW(gen.parameters(), lr=lr, weight_decay=0.01)

    def level_losses(ids: torch.Tensor, rate: float) -> list[torch.Tensor]:
        with torch.no_grad():
            codes, _, _ = tok.quantize(tok.encode(ids))
            noisy = [corrupt(c, rate) for c in codes]
            after = tok.codes_to_fhats(noisy)
            canvases = [torch.zeros_like(after[0])] + after[:-1]
        logits = gen(canvases)
        return [F.cross_entropy(lg.flatten(0, 1).float(), c.flatten())
                for lg, c in zip(logits, codes)]

    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        ids = train_data[torch.randint(len(train_data), (BATCH,))].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            losses = level_losses(ids, rate=torch.rand(1).item() * max_corrupt)
            loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()

        if step % 1000 == 0:
            gen.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=dev == "cuda"):
                val = level_losses(test_data[:512].to(dev), rate=0.0)
            gen.train()
            print(f"step {step:5d} loss {loss.item():.3f} "
                  f"val/level {[round(v.item(), 2) for v in val]}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), ART_DIR / "generator.pt")


if __name__ == "__main__":
    train()
