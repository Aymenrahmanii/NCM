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


def pick_tokens(logits: torch.Tensor, freq: float, mu: float, adj: list[float] | None,
                allowance: torch.Tensor | None) -> torch.Tensor:
    """Greedy decode with the official frequency/adjacency penalties (eval_shard.pick_tokens).

    Picks exactly the same tokens as upstream's loop, with less work per step: only one count
    per row changes each step, so the penalty and deficit terms are kept as running (b, v)
    tensors updated at the picked tokens instead of being rebuilt densely, and logits are
    converted to float64 one step at a time instead of all at once.
    """
    if freq <= 0 and not adj:
        return logits.argmax(-1)
    b, t_len, v = logits.shape
    dev, f64 = logits.device, torch.float64
    adj = adj or [0.0]
    al, aw, ad = adj[0], int(adj[1]) if len(adj) > 1 else 1, adj[2] if len(adj) > 2 else 1.0
    allow = (allowance[:v] * t_len)[None, :] if allowance is not None else None
    use_deficit = freq > 0 and mu > 0 and allow is not None

    cnt = torch.zeros(b, v, dtype=f64, device=dev)
    pen = torch.zeros(b, v, dtype=f64, device=dev)  # freq * excess, as upstream computes it
    deficit = allow.expand(b, v).clone() if use_deficit else None  # (allow - cnt).clamp(0)
    adj_vals = [torch.full((b, 1), -al * ad**j, dtype=f64, device=dev) for j in range(aw)]
    one = torch.ones(b, 1, dtype=f64, device=dev)
    out = torch.empty(b, t_len, dtype=torch.long, device=dev)
    hist: list[torch.Tensor] = []
    for t in range(t_len):
        step = logits[:, t, :].double()
        if freq > 0:
            step -= pen
            if use_deficit:
                step += mu * ((t_len - t) / t_len) * deficit
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], adj_vals[j])
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        idx = p[:, None]
        cnt.scatter_add_(1, idx, one)
        if freq > 0:
            c = cnt.gather(1, idx)
            a = allow.expand(b, v).gather(1, idx) if allow is not None else None
            pen.scatter_(1, idx, freq * ((c - a).clamp_min(0.0) if a is not None else c))
            if use_deficit:
                deficit.scatter_(1, idx, (a - c).clamp_min(0.0))
    return out


def _from_state(state: dict, spec, device: str) -> torch.Tensor | None:
    """A tensor named in the config as "generator.pt:inference.token_unigram" (float64)."""
    if not spec or ":" not in str(spec) or not state:
        return None
    obj = state
    for key in str(spec).split(":", 1)[1].split("."):
        obj = obj[key]
    return obj.double().to(device)


def _compile_cached_passes(net) -> None:
    """Compile the prefix pass and the per-level pass of the cached path with CUDA graphs.

    Upstream PR #5 keeps both methods free of side effects so they can be compiled. CUDA
    graphs reuse their output buffers on the next replay, but the prefix K/V must outlive the
    level passes that follow, so outputs are cloned as they leave the graph.
    """
    if not hasattr(net, "_kv_level"):
        raise RuntimeError("NCM_ROOT checkout lacks upstream PR #5 (_kv_prefix / _kv_level)")
    # _kv_level gets one graph per (level, batch shape): 10 levels x 4 shapes on WMT14. Past
    # dynamo's default limit of 8 it silently runs the remaining levels eagerly.
    import torch._dynamo.config as dynamo_config

    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 128)
    dynamo_config.fail_on_recompile_limit_hit = True  # a benchmark must not time eager code
    prefix =torch.compile(net._kv_prefix, mode="reduce-overhead", dynamic=False)
    level = torch.compile(net._kv_level, mode="reduce-overhead", dynamic=False)

    def kv_prefix(h, mask, rope):
        h, kv = prefix(h, mask, rope)
        return h.clone(), [(k.clone(), v.clone()) for k, v in kv]

    net._kv_prefix = kv_prefix
    net._kv_level = lambda x, kv, mask, rope, hidx: level(x, kv, mask, rope, hidx).clone()


class NCMModel:
    """Text-in, text-out wrapper for one conditional NCM checkpoint (xsum or wmt14)."""

    task: str = ""
    random_weights: bool = False  # build the architecture from the config, load no weights

    def __init__(self, device: str = "cuda", precision: str = "fp32", batch_size: int = 16,
                 checkpoint_dir: str | Path | None = None, kv_cache: bool = False,
                 cast_weights: bool = False, compiled: bool = False):
        # kv_cache=True is only correct with upstream's cache fix (PR #5): set NCM_ROOT to a
        # checkout that has it. On upstream main the cached path ignores prefix padding.
        # cast_weights=True stores the weights in `precision` (like the AR baselines'
        # torch_dtype); otherwise they stay fp32 and only the compute runs under autocast.
        from transformers import AutoTokenizer

        NextConceptTransformer, VQVAETokenizer = _official()
        ckpt = Path(checkpoint_dir or ARTIFACTS_DIR / self.task)
        cfg = self._load_config(ckpt)
        cfg.tokenizer.vqvae.checkpoint_path = str(ckpt / cfg.checkpoints.tokenizer)
        for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
            if cfg.checkpoints.get(name):  # TinyStories ships its own BPE files
                cfg.tokenizer.pre_tokenizer[key] = str(ckpt / cfg.checkpoints[name])

        self.device = device
        self.dtype = PRECISIONS[precision]
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.cfg = cfg
        if self.random_weights:
            import hydra
            from models.vqvae import VQVAE
            from utils.registry import tokenizers

            pre = hydra.utils.get_class(tokenizers[cfg.tokenizer.pre_tokenizer.name])
            self.tk = VQVAETokenizer(pre.from_config(cfg.tokenizer.pre_tokenizer),
                                     VQVAE(**cfg.tokenizer.vqvae))
        else:
            self.tk = VQVAETokenizer.from_config(cfg.tokenizer)
        self.tk.vqvae = self.tk.vqvae.to(device).eval()
        self.net = NextConceptTransformer(vqvae_local=self.tk.vqvae, config=cfg.model).to(device)
        state = {} if self.random_weights else torch.load(
            ckpt / cfg.checkpoints.generator, map_location="cpu", weights_only=False)
        if state:
            self.net.load_state_dict(state["model"], strict=True)
        self.net.eval()
        if cast_weights and self.dtype != torch.float32:
            self.net.to(self.dtype)
            self.tk.vqvae.to(self.dtype)
        self.compiled = compiled
        if compiled:
            if not kv_cache:
                raise ValueError("compiled=True compiles the KV-cached passes; set kv_cache=True")
            _compile_cached_passes(self.net)

        gen = cfg.generation
        self.temperature = float(gen.get("temperature", 1.0))
        self.top_k = int(gen.get("top_k", 1))
        self.top_p = float(gen.get("top_p", 0.0))
        self.cfg_scale = float(gen.get("cfg_scale", 0.0))
        self.freq = float(gen.get("frequency_penalty", 0.0))
        self.mu = float(gen.get("frequency_deficit", 0.0))
        adj = str(gen.get("adjacency_penalty", "") or "")
        self.adj = [float(v) for v in adj.split(",")] if adj else None
        self.allowance = _from_state(state, gen.get("frequency_allowance"), device)
        self.l0_probs = _from_state(state, gen.get("level0_unigram"), device)
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
        return pick_tokens(logits, self.freq, self.mu, self.adj, self.allowance)

    def _load_config(self, ckpt: Path):
        from omegaconf import OmegaConf

        return OmegaConf.load(ckpt / "config.yaml")

    def _decode_ids(self, ids: list[int]) -> str:
        if self.eos in ids:
            ids = ids[: ids.index(self.eos)]
        return self.tk.pre_tokenizer.decode([t for t in ids if t not in (self.eos, self.pad)],
                                            skip_special_tokens=True).strip()

    def _autocast(self):
        if self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type=torch.device(self.device).type, dtype=self.dtype)

    # ---- public API -------------------------------------------------------------------
    # prepare -> run -> finish, so the benchmark can time generation (run) on its own
    def prepare(self, items: list) -> torch.Tensor:
        return torch.tensor([self.prefix_ids(it) for it in items], device=self.device)

    @torch.no_grad()
    def run(self, x: torch.Tensor, return_levels: bool = False,
            l0: torch.Tensor | None = None):
        with self._autocast():
            codes = self.net.generate(l0, x, temperature=self.temperature, top_k=self.top_k,
                                      top_p=self.top_p, cfg_scale=self.cfg_scale,
                                      cfg_null_eos=self.eos, use_kv_cache=self.kv_cache)
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


class NCMUnconditional(NCMModel):
    """Unconditional samples: a released checkpoint (TinyStories, LM1B), or, with
    random_weights=True, the architecture of an unreleased one built from its upstream config.

    Follows eval_shard.py: the prefix is one EOS token and level 0 is drawn from the
    checkpoint's stored unigram (uniform when there are no weights). `prepare` takes any list
    and returns one sample per element. `length` (random weights only) keeps the levels below
    it and ends the schedule at `length`, so the OWT schedule 1, 4, 9, ..., 1024 can be timed
    at shorter canvases.
    """

    def __init__(self, task: str = "tinystories", random_weights: bool = False,
                 config_path: str | Path | None = None, length: int | None = None, **kwargs):
        self.task = task
        self.random_weights = random_weights
        self._config_path = Path(config_path) if config_path else None
        self._length = length
        super().__init__(**kwargs)
        self.canvas = int(self.tk.vqvae.vq.levels[-1])

    def _load_config(self, ckpt: Path):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(self._config_path or ckpt / "config.yaml")
        if self._length:
            if not self.random_weights:
                raise ValueError("length= changes the architecture; only with random_weights")
            _truncate_levels(cfg, self._length)
        return cfg

    def prepare(self, items: list):
        b = len(items)
        x = torch.full((b, 1), self.eos, device=self.device)
        n0 = int(self.tk.vqvae.vq.levels[0])
        probs = self.l0_probs.float() if self.l0_probs is not None else torch.ones(
            self.tk.vqvae.vq.embeddings[0].num_embeddings, device=self.device)
        return x, torch.multinomial(probs, b * n0, replacement=True).view(b, n0)

    def run(self, prepared, return_levels: bool = False):
        x, l0 = prepared
        return super().run(x, return_levels=return_levels, l0=l0)


def _truncate_levels(cfg, length: int) -> None:
    """Keep the levels below `length` and end at `length` (both tokenizer and generator)."""
    vq = cfg.tokenizer.vqvae
    levels = [int(v) for v in vq.vector_quantizer_config.levels.manual_levels if v < length]
    levels.append(length)
    n = len(levels)
    vq.vector_quantizer_config.levels.manual_levels = levels
    vq.vqvae_vocab_size = list(vq.vqvae_vocab_size)[:n]
    vq.context_length = length
    cfg.model.vocab_size = list(cfg.model.vocab_size)[: n - 1]
    cfg.model.context_length = length + 1


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
