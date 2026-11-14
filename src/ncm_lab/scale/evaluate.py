"""Phase 10 evaluation: temperature tuning on val, quality on test, latency, Pareto plot.

    python -m ncm_lab.scale.evaluate quality      # tune T on val by MAUVE, then score test
    python -m ncm_lab.scale.evaluate latency      # batch-1 time per 256-token sample
    python -m ncm_lab.scale.evaluate latency --device cpu
    python -m ncm_lab.scale.evaluate plot

Every finished run in artifacts/scale/ (except the tokenizer) is evaluated, one sampling
setting per model: the temperature from TEMPS with the best MAUVE against validation
stories (N_TUNE samples). On test, N_TEST samples get GPT-2-large perplexity, MAUVE against
5 disjoint draws of test stories, distinct-2 and unigram entropy (GPT-2 tokens). References
are real stories cut to 256 BPE tokens, exactly like the training data. Run it when no
training job holds the GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ncm_lab.toy_ncm.data import decode_ids, load_bpe
from ncm_lab.toy_ncm.evaluate import distinct_2

from .data import load_split
from .models import ARConfig, ARModel, Generator, GeneratorConfig, ncm_sample
from .train import ART, load_tokenizer

RESULTS = Path("results")
TEMPS = (0.6, 0.7, 0.8, 0.9, 1.0)
N_TUNE, N_TEST, DRAWS = 256, 512, 5
BATCH = 128
JUDGE = "gpt2-large"


def finished_runs() -> dict[str, dict]:
    runs = {}
    for path in sorted(ART.glob("*/model.pt")):
        meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
        if meta["kind"] != "tokenizer":
            runs[path.parent.name] = meta
    return runs


class Sampler:
    """Loads one run and turns (n, temperature, seed) into decoded stories."""

    def __init__(self, name: str, dev: str):
        ck = torch.load(ART / name / "model.pt", map_location=dev, weights_only=False)
        self.kind, self.dev = ck["meta"]["kind"], dev
        if self.kind == "gen":
            self.tok = load_tokenizer(dev)
            self.model = Generator(GeneratorConfig(**ck["meta"]["config"]), self.tok.cfg)
        else:
            self.model = ARModel(ARConfig(**ck["meta"]["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.to(dev).eval()

    @torch.no_grad()
    def ids(self, batch: int, temperature: float | list[float]) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.dev == "cuda"):
            if self.kind == "gen":
                _, canvases = ncm_sample(self.model, self.tok, batch, temperature)
                return self.tok.decode(canvases[-1]).argmax(-1)
            return self.model.generate(batch, temperature)

    def texts(self, n: int, temperature, seed: int, bpe) -> list[str]:
        torch.manual_seed(seed)
        out = []
        for i in range(0, n, BATCH):
            out += [decode_ids(bpe, r.tolist()) for r in self.ids(min(BATCH, n - i), temperature)]
        return out


class Judge:
    """GPT-2 large, loaded once: MAUVE features (last-token hidden state) and perplexity."""

    def __init__(self, dev: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(JUDGE)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(JUDGE).to(dev).eval()
        self.dev = dev

    def _batches(self, texts: list[str], bos: bool, size: int = 16):
        start = self.tok.bos_token if bos else ""  # MAUVE's featurizer uses no BOS
        for i in range(0, len(texts), size):
            chunk = [start + (t or ".") for t in texts[i:i + size]]
            yield self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=400).to(self.dev)

    @torch.no_grad()
    def features(self, texts: list[str]) -> np.ndarray:
        feats = []
        for enc in self._batches(texts, bos=False):
            h = self.model(**enc, output_hidden_states=True).hidden_states[-1]
            last = enc.attention_mask.sum(1) - 1
            feats.append(h[torch.arange(len(h)), last].float().cpu())
        return torch.cat(feats).numpy()

    @torch.no_grad()
    def perplexity(self, texts: list[str]) -> float:
        nll, n = 0.0, 0
        for enc in self._batches(texts, bos=True):
            target = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)[:, 1:]
            logits = self.model(**enc).logits[:, :-1]
            nll += F.cross_entropy(logits.flatten(0, 1).float(), target.flatten(),
                                   ignore_index=-100, reduction="sum").item()
            n += (target != -100).sum().item()
        return math.exp(nll / n)

    def entropy(self, texts: list[str]) -> float:
        """Mean per-sample unigram entropy in nats over GPT-2 tokens (as in Phase 9)."""
        vals = []
        for t in texts:
            counts = Counter(self.tok.encode(t))
            if counts:
                p = np.array(list(counts.values())) / sum(counts.values())
                vals.append(float(-(p * np.log(p)).sum()))
        return float(np.mean(vals))


def mauve(p: np.ndarray, q: np.ndarray) -> float:
    from mauve import compute_mauve

    return float(compute_mauve(p_features=p, q_features=q, seed=25, verbose=False).mauve)


def reference_texts(split: str, n: int, bpe, seed: int = 0) -> list[str]:
    data = load_split(split)
    rows = np.random.default_rng(seed).permutation(len(data))[:n]
    return [decode_ids(bpe, data[i].tolist()) for i in rows]


def quality(dev: str) -> None:
    bpe = load_bpe()
    judge = Judge(dev)
    val_ref = judge.features(reference_texts("val", N_TUNE, bpe))
    test_refs = reference_texts("test", N_TEST * DRAWS, bpe)
    test_feats = [judge.features(test_refs[d * N_TEST:(d + 1) * N_TEST]) for d in range(DRAWS)]
    tune_rows, rows = [], []
    human = test_refs[:N_TEST]
    rows.append(_score("human (test stories)", "", None, human, test_feats[0], test_feats[1:],
                       judge))
    samples = open(RESULTS / "phase10_samples.jsonl", "w", encoding="utf-8")
    for name, meta in finished_runs().items():
        s = Sampler(name, dev)
        best = (-1.0, None)
        for t in TEMPS:
            m = mauve(judge.features(s.texts(N_TUNE, t, seed=1, bpe=bpe)), val_ref)
            tune_rows.append({"run": name, "temperature": t, "val_mauve": round(m, 4)})
            print(tune_rows[-1], flush=True)
            best = max(best, (m, t))
        texts = s.texts(N_TEST, best[1], seed=2, bpe=bpe)
        for t in texts:
            samples.write(json.dumps({"run": name, "temperature": best[1], "text": t}) + "\n")
        feats = judge.features(texts)
        rows.append(_score(name, meta["kind"], best[1], texts, feats, test_feats, judge))
        del s
        torch.cuda.empty_cache()
    samples.close()
    _write(RESULTS / "phase10_tune.csv", tune_rows)
    _write(RESULTS / "phase10_quality.csv", rows)


def _score(name: str, kind: str, temp, texts: list[str], feats: np.ndarray,
           targets: list[np.ndarray], judge: Judge) -> dict:
    scores = [mauve(feats, r) for r in targets]
    row = {"run": name, "kind": kind, "temperature": temp, "n": len(texts),
           "gpt2_large_ppl": round(judge.perplexity(texts), 2),
           "mauve": round(float(np.mean(scores)), 4),
           "mauve_sd": round(float(np.std(scores, ddof=1)), 4),
           "distinct_2": round(distinct_2(texts), 4), "entropy": round(judge.entropy(texts), 3),
           "words": round(float(np.mean([len(t.split()) for t in texts])), 1)}
    print(row, flush=True)
    return row


def _sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def latency(dev: str, repeats: int) -> None:
    """Median seconds for one 256-token sample at batch 1, models interleaved per repeat."""
    from ncm_lab.toy_ncm.modules import count_params

    samplers = {name: Sampler(name, dev) for name in finished_runs()}
    temps = {r["run"]: float(r["temperature"]) for r in _read(RESULTS / "phase10_quality.csv")
             if r["kind"]} if (RESULTS / "phase10_quality.csv").exists() else {}
    times: dict[str, list[float]] = {n: [] for n in samplers}
    for r in range(repeats + 3):
        for name, s in samplers.items():
            _sync(dev)
            t0 = time.perf_counter()
            s.ids(1, temps.get(name, 1.0))
            _sync(dev)
            if r >= 3:  # warm-up
                times[name].append(time.perf_counter() - t0)
    rows = []
    for name, s in samplers.items():
        params = count_params(s.model) + (count_params(s.tok) if s.kind == "gen" else 0)
        ts = sorted(times[name])
        rows.append({"run": name, "device": dev, "params_m": round(params / 1e6, 2),
                     "median_ms": round(1000 * statistics.median(ts), 2),
                     "p25_ms": round(1000 * ts[len(ts) // 4], 2),
                     "p75_ms": round(1000 * ts[3 * len(ts) // 4], 2), "repeats": len(ts)})
        print(rows[-1], flush=True)
    _write(RESULTS / f"phase10_latency_{dev}.csv", rows)


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = {r["run"]: r for r in _read(RESULTS / "phase10_quality.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in ((axes[0], "mauve", "MAUVE vs test stories (higher better)"),
                              (axes[1], "gpt2_large_ppl", "GPT-2 large perplexity (lower better)")):
        for dev, marker in (("cuda", "o"), ("cpu", "s")):
            path = RESULTS / f"phase10_latency_{dev}.csv"
            if not path.exists():
                continue
            for r in _read(path):
                if r["run"] not in q:
                    continue
                is_ncm = r["run"].startswith("gen")
                ax.scatter(float(r["median_ms"]), float(q[r["run"]][metric]), marker=marker,
                           color="tab:red" if is_ncm else "tab:blue")
                ax.annotate(r["run"], (float(r["median_ms"]), float(q[r["run"]][metric])),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        if "human (test stories)" in q:
            ax.axhline(float(q["human (test stories)"][metric]), color="gray", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("batch-1 latency per 256-token sample (ms; o GPU, s CPU)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].set_yscale("log")
    fig.suptitle("Phase 10: NCM (red) vs AR ladder (blue), same data and BPE")
    fig.tight_layout()
    fig.savefig(RESULTS / "phase10_pareto.png", dpi=130)


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["quality", "latency", "plot"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args(argv)
    if a.cmd == "quality":
        quality(a.device)
    elif a.cmd == "latency":
        latency(a.device, a.repeats)
    else:
        plot()


if __name__ == "__main__":
    main()
