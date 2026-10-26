"""Phase 3: clean wrappers around the official NCM checkpoints.

    from ncm_lab.models.ncm import NCMSummarizer, NCMTranslator
    NCMSummarizer().generate(["Long news article ..."])     # -> ["One-sentence summary."]
    NCMTranslator().generate(["Guten Morgen, wie geht's?"])  # -> ["Good morning, how are you?"]

The official code lives in third_party/ncm (cloned by scripts/download_ncm.sh) and the weights
in artifacts/<task>/. Generation logic follows third_party/ncm/scripts/eval_shard.py.
"""

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

NCM_ROOT = Path(os.environ.get("NCM_ROOT", "third_party/ncm")).resolve()
ARTIFACTS_DIR = Path(os.environ.get("NCM_ARTIFACTS_DIR", "artifacts")).resolve()

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _official():
    """Import the official classes (they use top-level packages like `models`)."""
    if str(NCM_ROOT) not in sys.path:
        sys.path.insert(0, str(NCM_ROOT))
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    return NextConceptTransformer, VQVAETokenizer


class NCMModel:
    """Text-in, text-out wrapper for one conditional NCM checkpoint (xsum or wmt14)."""

    task: str = ""

    def __init__(self, device: str = "cuda", precision: str = "fp32", batch_size: int = 16,
                 checkpoint_dir: str | Path | None = None):
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        NextConceptTransformer, VQVAETokenizer = _official()
        ckpt = Path(checkpoint_dir or ARTIFACTS_DIR / self.task)
        cfg = OmegaConf.load(ckpt / "config.yaml")
        cfg.tokenizer.vqvae.checkpoint_path = str(ckpt / cfg.checkpoints.tokenizer)

        self.device = device
        self.dtype = PRECISIONS[precision]
        self.batch_size = batch_size
        self.cfg = cfg
        self.tk = VQVAETokenizer.from_config(cfg.tokenizer)
        self.tk.vqvae = self.tk.vqvae.to(device).eval()
        self.net = NextConceptTransformer(vqvae_local=self.tk.vqvae, config=cfg.model).to(device)
        state = torch.load(ckpt / cfg.checkpoints.generator, map_location="cpu", weights_only=False)
        self.net.load_state_dict(state["model"], strict=True)
        self.net.eval()

        gen = cfg.generation
        self.temperature = float(gen.get("temperature", 1.0))
        self.top_k = int(gen.get("top_k", 1))
        self.top_p = float(gen.get("top_p", 0.0))
        self.cfg_scale = float(gen.get("cfg_scale", 0.0))
        self.freq = float(gen.get("frequency_penalty", 0.0))
        self.mu = float(gen.get("frequency_deficit", 0.0))
        adj = str(gen.get("adjacency_penalty", "") or "")
        self.adj = [float(v) for v in adj.split(",")] if adj else None
        self.allowance = None
        spec = gen.get("frequency_allowance")
        if spec and ":" in str(spec):  # e.g. "generator.pt:inference.token_unigram"
            obj = state
            for key in str(spec).split(":", 1)[1].split("."):
                obj = obj[key]
            self.allowance = obj.double().to(device)
        del state

        self.prefix_len = int(cfg.model.prefix_len)
        self.eos = self.tk.pre_tokenizer.eos_token_id
        self.pad = self.tk.pre_tokenizer.pad_token_id
        # raw text -> ids with the same Hugging Face tokenizer the model was trained on
        model_id = cfg.tokenizer.pre_tokenizer.get("model_id")
        self.src_tok = AutoTokenizer.from_pretrained(str(model_id)) if model_id else None

    # ---- input side -------------------------------------------------------------------
    def prefix_ids(self, item: str | list[int]) -> list[int]:
        """Raw text (or already-tokenized ids) -> fixed-length prefix padded with PAD."""
        ids = self.src_tok(item)["input_ids"] if isinstance(item, str) else list(item)
        return (ids[: self.prefix_len] + [self.pad] * self.prefix_len)[: self.prefix_len]

    # ---- output side ------------------------------------------------------------------
    def _pick_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy decode, with the official frequency/adjacency penalties when configured."""
        if self.freq <= 0 and not self.adj:
            return logits.argmax(-1)
        b, t_len, v = logits.shape
        lg = logits.double()
        allow = (self.allowance[:v] * t_len)[None, :] if self.allowance is not None else None
        adj = self.adj or [0.0]
        al, aw, ad = adj[0], int(adj[1]) if len(adj) > 1 else 1, adj[2] if len(adj) > 2 else 1.0
        cnt = torch.zeros(b, v, dtype=torch.float64, device=lg.device)
        out = torch.empty(b, t_len, dtype=torch.long, device=lg.device)
        hist = []
        for t in range(t_len):
            step = lg[:, t, :].clone()
            if self.freq > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= self.freq * excess
                if self.mu > 0 and allow is not None:
                    step += self.mu * ((t_len - t) / t_len) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((b, 1), -al * ad**j,
                                                            dtype=torch.float64, device=lg.device))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(b, 1, dtype=torch.float64, device=lg.device))
        return out

    def _decode_ids(self, ids: list[int]) -> str:
        if self.eos in ids:
            ids = ids[: ids.index(self.eos)]
        return self.tk.pre_tokenizer.decode([t for t in ids if t not in (self.eos, self.pad)],
                                            skip_special_tokens=True).strip()

    def _autocast(self):
        if self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    # ---- public API -------------------------------------------------------------------
    # prepare -> run -> finish, so the benchmark can time generation (run) on its own
    def prepare(self, items: list) -> torch.Tensor:
        return torch.tensor([self.prefix_ids(it) for it in items], device=self.device)

    @torch.no_grad()
    def run(self, x: torch.Tensor, return_levels: bool = False):
        with self._autocast():
            codes = self.net.generate(None, x, temperature=self.temperature, top_k=self.top_k,
                                      top_p=self.top_p, cfg_scale=self.cfg_scale,
                                      cfg_null_eos=self.eos)
            _, z_levels = self.tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            tokens = self._pick_tokens(self.tk.vqvae.decoder(z_levels[-1]).float())
            if not return_levels:
                return tokens
            levels = [self.tk.vqvae.decoder(z).float().argmax(-1) for z in z_levels]
        return tokens, levels

    def finish(self, tokens: torch.Tensor) -> list[str]:
        return [self._decode_ids(r) for r in tokens.tolist()]

    def generate_batch(self, items: list) -> list[str]:
        return self.finish(self.run(self.prepare(items)))

    def generate(self, texts: list) -> list[str]:
        """One output string per input (raw strings or pre-tokenized id lists)."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            out += self.generate_batch(texts[i : i + self.batch_size])
        return out

    def generate_levels(self, text: str) -> list[str]:
        """Decoded text after each level, coarse to fine (the blog's level-by-level demo)."""
        _, levels = self.run(self.prepare([text]), return_levels=True)
        return [self.finish(lv)[0] for lv in levels]

    @property
    def n_params(self) -> int:
        """Generator only (the number the paper reports)."""
        return sum(p.numel() for p in self.net.parameters())

    @property
    def n_params_total(self) -> int:
        """Generator + VQ-VAE tokenizer, both needed at inference."""
        return self.n_params + sum(p.numel() for p in self.tk.vqvae.parameters())


class NCMSummarizer(NCMModel):
    """News article -> one-sentence summary (trained on XSum, BBC news)."""

    task = "xsum"


class NCMTranslator(NCMModel):
    """German -> English (trained on WMT14)."""

    task = "wmt14"


class NCMSudokuSolver(NCMModel):
    """81-character puzzle (0 or . for blanks) -> 81-digit solution."""

    def __init__(self, difficulty: str = "hard", **kwargs):
        self.task = f"sudoku/{difficulty}"
        super().__init__(**kwargs)

    def prefix_ids(self, item: str | list[int]) -> list[int]:
        if not isinstance(item, str):
            return list(item)
        cells = [c for c in item if c in "0123456789."]
        if len(cells) != 81:
            raise ValueError(f"expected 81 cells, got {len(cells)}")
        bos, sep = 11, 10  # see third_party/ncm/tokenization/sudoku_tokenizer.py
        ids = [bos]
        for r in range(9):
            ids += [0 if c in "0." else int(c) for c in cells[9 * r : 9 * r + 9]]
            if r < 8:
                ids.append(sep)
        return ids + [bos]

    def finish(self, tokens: torch.Tensor) -> list[str]:
        return [s.replace("|", "") for s in super().finish(tokens)]

    @staticmethod
    def is_valid(solution: str, puzzle: str | None = None) -> bool:
        """Every row, column and box holds 1-9, and the given clues are kept."""
        if len(solution) != 81 or not solution.isdigit() or "0" in solution:
            return False
        rows = [solution[9 * i : 9 * i + 9] for i in range(9)]
        cols = ["".join(r[j] for r in rows) for j in range(9)]
        boxes = ["".join(rows[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
                 for br in (0, 3, 6) for bc in (0, 3, 6)]
        if not all(sorted(x) == list("123456789") for x in rows + cols + boxes):
            return False
        if puzzle is not None:
            clues = [c for c in puzzle if c in "0123456789."]
            return all(p in "0." or p == s for p, s in zip(clues, solution))
        return True
