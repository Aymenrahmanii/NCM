"""Phase 9: does NCM win at batch 1 when outputs are long?

    NCM_ROOT=third_party/ncm-pr5-pr4 python -m ncm_lab.bench.long_outputs quality
    NCM_ROOT=third_party/ncm-pr5-pr4 python -m ncm_lab.bench.long_outputs speed --length 128
    NCM_ROOT=third_party/ncm-pr5-pr4 python -m ncm_lab.bench.long_outputs speed --length 1024

The 1,024-token OpenWebText checkpoint the paper times is not released (its Hub folder holds
only a TODO), and every released unconditional checkpoint has a 128-token canvas. So:

- quality (128 tokens, real weights): NCM TinyStories against TinyStories-33M (in-domain AR)
  and GPT-2 small (out of domain here). Upstream's
  protocol: 5 seeds x 256 samples, generative perplexity under GPT-2 large. Unigram entropy
  is reported on GPT-2 tokens for every model, so it is comparable across tokenizers, and on
  NCM's own tokens as upstream reports it.
- speed: at 128 tokens the real TinyStories NCM; at longer lengths the OWT architecture from
  upstream's config/eval/owt.yaml with random weights (latency does not depend on the
  weights), its level schedule cut at the length, against GPT-2 small and GPT-2 medium (the
  latter matches its 317M parameters). AR models write exactly that many tokens.
  Interleaved protocol from ncm_lab.bench.interleaved. Everything runs in bf16: fp16
  overflows in the sampling NCM checkpoints.

`mauve` scores the saved quality samples against human TinyStories (MAUVE, GPT-2 large
features; needs data/tinystories_text/TinyStories-valid.txt from roneneldan/TinyStories).

Writes results/phase9_quality.csv (+ _samples.jsonl), results/phase9_mauve.csv and
results/phase9_speed_<length>{.csv,_ratios.csv,_calls.jsonl}.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ncm_lab.bench.interleaved import run_batch_size, write_results
from ncm_lab.stats import bootstrap_quantile, paired_ratio_of_medians

RESULTS = Path("results")
PRECISION = "bf16"
SEEDS = (42, 0, 1, 2, 3)  # upstream config/eval/tinystories.yaml
OWT_CONFIG = Path("third_party/ncm-pr5-pr4/config/eval/owt.yaml")
AR_IDS = {"ar-tinystories": "roneneldan/TinyStories-33M", "ar-gpt2": "gpt2",
          "ar-gpt2-medium": "gpt2-medium"}
QUALITY_MODELS = ("ncm", "ar-tinystories", "ar-gpt2")


def make_ncm(length: int, compiled: bool):
    from ncm_lab.models.ncm import NCMUnconditional

    kw = dict(precision=PRECISION, kv_cache=True, cast_weights=True, compiled=compiled)
    if length == 128:
        return NCMUnconditional("tinystories", **kw)
    return NCMUnconditional("owt", random_weights=True, length=length, config_path=OWT_CONFIG,
                            checkpoint_dir="artifacts/owt", **kw)


def make_ar(name: str, length: int, forced: bool, compiled: bool):
    from ncm_lab.models.baselines import HFCausalLM

    return HFCausalLM(AR_IDS[name], length=length, precision=PRECISION, forced=forced,
                      compiled=compiled)


def entropy(ids: list[int]) -> float:
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def sample(model, n: int, bs: int, seed: int) -> list[str]:
    torch.manual_seed(seed)
    out = []
    for i in range(0, n, bs):
        out += model.finish(model.run(model.prepare(list(range(min(bs, n - i))))))
    return out


def quality(args) -> None:
    import gc
    import os
    import sys

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, os.environ.get("NCM_ROOT", "third_party/ncm"))
    from evaluation.gen_ppl import generation_perplexity

    texts: dict[str, dict[int, list[str]]] = {}
    native: dict[str, list[float]] = {}
    for name in QUALITY_MODELS:
        print(f"[quality] sampling {name}", flush=True)
        m = make_ncm(128, compiled=False) if name == "ncm" else make_ar(name, 128, False, False)
        texts[name] = {s: sample(m, args.n, args.bs, s) for s in SEEDS}
        if name == "ncm":
            enc = m.tk.pre_tokenizer.encode
            native[name] = [float(np.mean([entropy(enc(t)) for t in texts[name][s] if t]))
                            for s in SEEDS]
        del m
        gc.collect()
        torch.cuda.empty_cache()

    ref_tok = AutoTokenizer.from_pretrained("gpt2-large")
    ref = AutoModelForCausalLM.from_pretrained("gpt2-large").cuda().eval()
    rows = []
    with open(RESULTS / "phase9_quality_samples.jsonl", "w", encoding="utf-8") as f:
        for name, by_seed in texts.items():
            ppl, ent, words = [], [], []
            for s, gen in by_seed.items():
                for t in gen:
                    f.write(json.dumps({"model": name, "seed": s, "text": t},
                                       ensure_ascii=False) + "\n")
                with torch.no_grad():
                    ppl.append(generation_perplexity(ref, ref_tok, [""] * len(gen), gen,
                                                     batch_size=32).mean_ppl)
                ent.append(float(np.mean([entropy(ref_tok.encode(t)) for t in gen if t])))
                words.append(float(np.mean([len(t.split()) for t in gen])))
            g, e = np.array(ppl), np.array(ent)
            row = {"model": name, "model_id": AR_IDS.get(name, "hazyresearch/ncm tinystories"),
                   "samples": sum(len(v) for v in by_seed.values()), "seeds": len(by_seed),
                   "gen_ppl": round(float(g.mean()), 2),
                   "gen_ppl_sd": round(float(g.std(ddof=1)), 2),
                   "entropy_gpt2_tokens": round(float(e.mean()), 3),
                   "entropy_gpt2_sd": round(float(e.std(ddof=1)), 3),
                   "entropy_native": round(float(np.mean(native[name])), 3) if name in native
                   else "", "words": round(float(np.mean(words)), 1)}
            print(row, flush=True)
            rows.append(row)
    with open(RESULTS / "phase9_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


REFERENCE = Path("data/tinystories_text/TinyStories-valid.txt")  # roneneldan/TinyStories


def mauve_scores(args) -> None:
    """MAUVE of each model's pooled quality samples against human TinyStories openings.

    References are cut to 128 GPT-2 tokens, like the samples. Each model is scored against
    5 disjoint draws of references; "human" scores draw 0 against draws 1-4 (the ceiling).
    """
    from mauve import compute_mauve
    from mauve.compute_mauve import get_features_from_input
    from transformers import AutoTokenizer

    by_model: dict[str, list[str]] = {}
    with open(RESULTS / "phase9_quality_samples.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_model.setdefault(r["model"], []).append(r["text"])
    n = min(len(v) for v in by_model.values())
    tok = AutoTokenizer.from_pretrained("gpt2")
    stories = [t.strip() for t in REFERENCE.read_text(encoding="utf-8").split("<|endoftext|>")]
    stories = [t for t in stories if t]
    order = np.random.default_rng(0).permutation(len(stories))[: n * args.draws]
    refs = [tok.decode(tok.encode(stories[i])[:128]).strip() for i in order]

    def feats(name: str, texts: list[str]):
        print(f"[mauve] featurizing {name} ({len(texts)})", flush=True)
        return get_features_from_input(None, None, texts, "gpt2-large", 256, 0, name,
                                       batch_size=16)

    ref_feats = [feats(f"ref{d}", refs[d * n:(d + 1) * n]) for d in range(args.draws)]
    sets = {"human": ref_feats[0]}
    sets.update({m: feats(m, t[:n]) for m, t in by_model.items()})
    rows = []
    for name, f in sets.items():
        targets = ref_feats[1:] if name == "human" else ref_feats
        scores = [compute_mauve(p_features=f, q_features=r, seed=25).mauve for r in targets]
        row = {"model": name, "n": n, "draws": len(scores),
               "mauve": round(float(np.mean(scores)), 4),
               "mauve_sd": round(float(np.std(scores, ddof=1)), 4)}
        print(row, flush=True)
        rows.append(row)
    with open(RESULTS / "phase9_mauve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def speed(args) -> None:
    L = args.length
    models = {"ncm": make_ncm(L, compiled=args.compiled)}
    # beyond 128 tokens NCM is the 317M OWT architecture: GPT-2 medium (355M) matches its size
    ar_names = ["ar-tinystories", "ar-gpt2"] if L == 128 else ["ar-gpt2", "ar-gpt2-medium"]
    for name in ar_names:
        models[name] = make_ar(name, L, forced=True, compiled=args.compiled)
    n_params = {k: round((getattr(m, "n_params_total", None) or m.n_params) / 1e6, 1)
                for k, m in models.items()}
    # Unconditional inputs all have the same shape, so 10 distinct batches are plenty and keep
    # the compiled models' warm-up (one call per distinct batch) short.
    items = list(range(10 * max(args.batch_sizes)))
    inputs = {name: items for name in models}
    stem = RESULTS / f"phase9_speed_{L}{'' if args.compiled else '_eager'}"
    rows, ratio_rows = [], []
    log: list[dict] = [{"length": L, "params_M": n_params, "compiled": args.compiled,
                        "torch": torch.__version__, "device": torch.cuda.get_device_name()}]
    for bs in args.batch_sizes:
        calls = args.calls if bs == 1 else args.calls_batched
        print(f"[speed] length {L} batch {bs}: {calls} rounds", flush=True)
        torch.manual_seed(0)
        times = run_batch_size(models, inputs, bs, calls, args.warmup, log)
        for name, t in times.items():
            med, (lo, hi) = bootstrap_quantile(t, 0.5)
            p95, _ = bootstrap_quantile(t, 0.95)
            rows.append({"length": L, "model": name, "params_M": n_params[name],
                         "compiled": args.compiled, "batch_size": bs, "calls": len(t),
                         "median_ms": round(med, 1), "median_lo": round(lo, 1),
                         "median_hi": round(hi, 1), "p95_ms": round(p95, 1),
                         "ms_per_token": round(med / L, 3)})
        n_before = len(ratio_rows)
        for a, b in itertools.combinations(models, 2):
            r, (lo, hi) = paired_ratio_of_medians(times[a], times[b])
            ratio_rows.append({"length": L, "batch_size": bs, "model_a": a, "model_b": b,
                               "median_latency_ratio_a_over_b": round(r, 3),
                               "lo": round(lo, 3), "hi": round(hi, 3),
                               "differs": not lo <= 1.0 <= hi})
        for row in rows[-len(times):] + ratio_rows[n_before:]:
            print(row, flush=True)
        write_results(stem, rows, ratio_rows, log)


LABELS = {"ncm": "NCM", "ar-tinystories": "TinyStories-33M", "ar-gpt2": "GPT-2 small",
          "ar-gpt2-medium": "GPT-2 medium"}
# The card had < 0.1 GB headroom here, and the driver spilled to system memory.
SPILLED = {(1024, 8)}


def plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for p in sorted(RESULTS.glob("phase9_speed_*[0-9].csv"))
            for r in csv.DictReader(open(p))]
    sizes = sorted({int(r["batch_size"]) for r in rows})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4), sharey=True)
    for ax, bs in zip(np.atleast_1d(axes), sizes):
        for name, label in LABELS.items():
            pts = sorted((int(r["length"]), float(r["median_ms"])) for r in rows
                         if r["model"] == name and int(r["batch_size"]) == bs)
            if not pts:
                continue
            ok = [(n, t) for n, t in pts if (n, bs) not in SPILLED]
            line, = ax.plot(*zip(*ok), marker="o", label=label)
            bad = [(n, t) for n, t in pts if (n, bs) in SPILLED]
            if bad:
                ax.plot(*zip(*bad), marker="x", ls="none", color=line.get_color())
        ax.set(xscale="log", yscale="log", xlabel="output tokens", title=f"batch {bs}")
        ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median ms per call (compiled, bf16)")
    axes[0].legend()
    fig.text(0.5, -0.02, "NCM: TinyStories checkpoint at 128, OWT architecture with random "
             "weights above. x = memory spill on the 6 GB card (not a valid timing).",
             ha="center", fontsize=8)
    fig.tight_layout()
    out = RESULTS / "phase9_latency_vs_length.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--n", type=int, default=256, help="samples per seed")
    q.add_argument("--bs", type=int, default=32)
    mv = sub.add_parser("mauve")
    mv.add_argument("--draws", type=int, default=5, help="disjoint reference draws")
    s = sub.add_parser("speed")
    s.add_argument("--length", type=int, required=True)
    s.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    s.add_argument("--calls", type=int, default=100)
    s.add_argument("--calls-batched", type=int, default=50)
    s.add_argument("--warmup", type=int, default=10)
    s.add_argument("--eager", dest="compiled", action="store_false")
    sub.add_parser("plot", help="latency against output length, from the speed CSVs")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    {"quality": quality, "mauve": mauve_scores, "speed": speed, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    main()
