"""Stage 1: multiscale residual VQ tokenizer.

Text (64 tokens) -> encoder -> latent z (64 x CODE_DIM).
Level k averages the residual (what earlier levels missed) down to LEVELS[k] positions,
snaps each position to the nearest code of its own codebook, and stretches it back to 64.
The decoder turns the sum of levels back into tokens.

    python -m ncm_lab.toy_ncm.tokenizer
"""

import csv
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (ART_DIR, BATCH, CODE_DIM, DIM, HEADS, LEVELS, N_CODES, PAD_ID,
                     RESULTS_DIR, SEQ_LEN, VOCAB_SIZE, device)
from .data import load_tokens
from .modules import Transformer, cosine_lr, count_params


def down(x: torch.Tensor, length: int) -> torch.Tensor:
    """(B, T, d) -> (B, length, d) by averaging neighbouring positions."""
    if x.shape[1] == length:
        return x
    return F.adaptive_avg_pool1d(x.transpose(1, 2), length).transpose(1, 2)


def up(x: torch.Tensor, length: int = SEQ_LEN) -> torch.Tensor:
    """(B, L, d) -> (B, length, d) by linear interpolation."""
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear",
                         align_corners=False).transpose(1, 2)


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, dim) * 0.02)
        self.beta = beta
        self.register_buffer("usage", torch.full((n_codes,), 1.0 / n_codes))
        self.register_buffer("ready", torch.tensor(False))
        self.recent: torch.Tensor | None = None

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.codebook)

    def forward(self, r: torch.Tensor):
        flat = r.reshape(-1, r.shape[-1])
        if self.training:
            if not self.ready:  # start codes on real data instead of noise
                self._reset(torch.ones_like(self.usage, dtype=torch.bool), flat.detach())
                self.ready.fill_(True)
            self.recent = flat.detach()
        cb = self.codebook
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(1)
        idx = dist.argmin(1)
        q = self.lookup(idx).view_as(r)
        if self.training:
            counts = torch.bincount(idx, minlength=len(self.usage)).float()
            self.usage.mul_(0.99).add_(0.01 * counts / counts.sum())
        # codebook loss pulls codes to the data, commitment loss pulls the encoder to codes
        loss = F.mse_loss(q, r.detach()) + self.beta * F.mse_loss(r, q.detach())
        return q, idx.view(r.shape[:-1]), loss

    @torch.no_grad()
    def _reset(self, dead: torch.Tensor, pool: torch.Tensor) -> None:
        n = int(dead.sum())
        pick = pool[torch.randint(len(pool), (n,), device=pool.device)]
        self.codebook.data[dead] = pick + 0.01 * pick.std() * torch.randn_like(pick)
        self.usage[dead] = 1.0 / len(self.usage)

    def restart_dead(self, threshold: float = 0.05) -> int:
        """Move codes that are almost never used onto recent data vectors."""
        dead = self.usage < threshold / len(self.usage)
        if dead.any() and self.recent is not None:
            self._reset(dead, self.recent)
        return int(dead.sum())


class MultiscaleTokenizer(nn.Module):
    def __init__(self, layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.enc_pos = nn.Parameter(torch.randn(1, SEQ_LEN, DIM) * 0.02)
        self.encoder = Transformer(DIM, HEADS, layers)
        self.to_code = nn.Linear(DIM, CODE_DIM)
        # fixed scale before quantizing, so the encoder can't drift away from the codebooks
        self.pre_quant_norm = nn.LayerNorm(CODE_DIM, elementwise_affine=False)
        self.quantizers = nn.ModuleList(VectorQuantizer(N_CODES, CODE_DIM) for _ in LEVELS)
        self.from_code = nn.Linear(CODE_DIM, DIM)
        self.dec_pos = nn.Parameter(torch.randn(1, SEQ_LEN, DIM) * 0.02)
        self.decoder = Transformer(DIM, HEADS, layers)
        self.head = nn.Linear(DIM, VOCAB_SIZE)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pre_quant_norm(self.to_code(self.encoder(self.embed(ids) + self.enc_pos)))

    def quantize(self, z: torch.Tensor):
        """Coarse-to-fine residual quantization.

        Returns codes per level, the canvas f_hat after each level, and the VQ loss.
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            f_hat = torch.zeros_like(z)
            codes, f_hats, loss = [], [], 0.0
            for length, vq in zip(LEVELS, self.quantizers):
                r = down(z - f_hat.detach(), length)    # what earlier levels missed
                q, idx, level_loss = vq(r)
                q = r + (q - r).detach()                # straight-through gradient
                f_hat = f_hat + up(q)
                codes.append(idx)
                f_hats.append(f_hat)
                loss = loss + level_loss
        return codes, f_hats, loss / len(LEVELS)

    def decode(self, f_hat: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(self.from_code(f_hat) + self.dec_pos))

    def codes_to_fhats(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Canvas after each level, rebuilt from discrete codes."""
        f_hat, out = 0.0, []
        for idx, vq in zip(codes, self.quantizers):
            f_hat = f_hat + up(vq.lookup(idx))
            out.append(f_hat)
        return out


@torch.no_grad()
def recon_accuracy(model: MultiscaleTokenizer, data: torch.Tensor, dev: str) -> list[float]:
    """Token accuracy (non-pad) when decoding from the first 1, 2, ... levels."""
    model.eval()
    hits = torch.zeros(len(LEVELS))
    total = 0
    for i in range(0, len(data), 256):
        ids = data[i : i + 256].to(dev)
        keep = ids != PAD_ID
        _, f_hats, _ = model.quantize(model.encode(ids))
        for k, f in enumerate(f_hats):
            hits[k] += ((model.decode(f).argmax(-1) == ids) & keep).sum().item()
        total += keep.sum().item()
    return (hits / total).tolist()


def load_tokenizer(dev: str) -> MultiscaleTokenizer:
    model = MultiscaleTokenizer().to(dev)
    model.load_state_dict(torch.load(ART_DIR / "tokenizer.pt", map_location=dev))
    model.eval().requires_grad_(False)
    return model


def train(steps: int = 8000, lr: float = 5e-4) -> None:
    torch.manual_seed(0)
    random.seed(0)
    dev = device()
    train_data, test_data = load_tokens()
    model = MultiscaleTokenizer().to(dev)
    print(f"tokenizer params {count_params(model) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n_levels = len(LEVELS)

    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr)
        ids = train_data[torch.randint(len(train_data), (BATCH,))].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            codes, f_hats, vq_loss = model.quantize(model.encode(ids))
            # nested dropout: half the batch decodes from a random prefix of the levels
            keep = torch.randint(1, n_levels + 1, (BATCH,), device=dev)
            keep = torch.where(torch.rand(BATCH, device=dev) < 0.5, n_levels, keep)
            f_sel = torch.stack(f_hats)[keep - 1, torch.arange(BATCH, device=dev)]
            logits = model.decode(f_sel)
            ce = F.cross_entropy(logits.flatten(0, 1).float(), ids.flatten(), ignore_index=PAD_ID)
            loss = ce + vq_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 and step < 0.8 * steps:
            dead = [vq.restart_dead() for vq in model.quantizers]
        if step % 1000 == 0:
            acc = recon_accuracy(model, test_data[:1000], dev)
            model.train()
            print(f"step {step:5d} ce {ce.item():.3f} vq {vq_loss.item():.4f} "
                  f"restarted {dead} acc/levels {[round(a, 3) for a in acc]}")

    ART_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ART_DIR / "tokenizer.pt")

    acc = recon_accuracy(model, test_data, dev)
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "phase2_recon_by_level.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["levels_used", "level_length", "token_accuracy"])
        for k, a in enumerate(acc):
            w.writerow([k + 1, LEVELS[k], round(a, 4)])
    print("test accuracy by levels used:", [round(a, 3) for a in acc])


if __name__ == "__main__":
    train()
