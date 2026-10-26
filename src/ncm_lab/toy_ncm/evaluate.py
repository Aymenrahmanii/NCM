"""Phase 2 evaluation: recon chart, level-by-level demo, sample quality and timing.

    python -m ncm_lab.toy_ncm.evaluate
"""

import csv
import math
import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .ar_baseline import load_ar  # noqa: E402
from .config import LEVELS, RESULTS_DIR, device  # noqa: E402
from .data import decode_ids, load_bpe, load_tokens  # noqa: E402
from .generator import load_generator, sample  # noqa: E402
from .modules import count_params  # noqa: E402
from .tokenizer import load_tokenizer, recon_accuracy  # noqa: E402

N_SAMPLES = 512
BATCH_SIZES = (1, 16, 128)


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def median_seconds(fn, dev: str, repeats: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        sync(dev)
        start = time.perf_counter()
        fn()
        sync(dev)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


@torch.no_grad()
def gpt2_perplexity(texts: list[str], dev: str) -> float:
    """Perplexity under GPT-2 small: a judge that neither model was trained with."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    nll, n = 0.0, 0
    for i in range(0, len(texts), 32):
        chunk = [tok.bos_token + (t or ".") for t in texts[i : i + 32]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
        target = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)[:, 1:]
        logits = model(**enc).logits[:, :-1]
        nll += F.cross_entropy(logits.flatten(0, 1).float(), target.flatten(),
                               ignore_index=-100, reduction="sum").item()
        n += (target != -100).sum().item()
    return math.exp(nll / n)


def distinct_2(texts: list[str]) -> float:
    bigrams = [tuple(w[i : i + 2]) for w in (t.split() for t in texts) for i in range(len(w) - 1)]
    return len(set(bigrams)) / max(1, len(bigrams))


def main() -> None:
    torch.manual_seed(0)
    dev = device()
    bpe = load_bpe()
    _, test = load_tokens()
    tok, gen, ar = load_tokenizer(dev), load_generator(dev), load_ar(dev)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    RESULTS_DIR.mkdir(exist_ok=True)
    report = ["# Phase 2 results (toy NCM vs autoregressive, RTX 3050)", ""]
    report.append(f"Params: tokenizer {count_params(tok) / 1e6:.2f}M, "
                  f"NCM generator {count_params(gen) / 1e6:.2f}M, AR {count_params(ar) / 1e6:.2f}M")

    # 1. reconstruction accuracy by number of levels
    acc = recon_accuracy(tok, test, dev)
    xs = list(range(1, len(LEVELS) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(xs, [a * 100 for a in acc], marker="o")
    plt.xticks(xs, [f"{k}\n(len {n})" for k, n in zip(xs, LEVELS)])
    plt.xlabel("levels used (cumulative)")
    plt.ylabel("token reconstruction accuracy (%)")
    plt.title("Toy NCM tokenizer: coarse to fine")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase2_recon_by_level.png", dpi=120)
    report += ["", "## Reconstruction accuracy by levels used", "",
               "| levels | length of last level | accuracy |", "|---|---|---|"]
    report += [f"| {k} | {n} | {a:.1%} |" for k, n, a in zip(xs, LEVELS, acc)]

    # 2. level-by-level printouts
    demo = []
    with torch.no_grad(), torch.autocast(**amp):
        ids = test[:2].to(dev)
        _, f_hats, _ = tok.quantize(tok.encode(ids))
        for j in range(2):
            demo += [f"== real story {j + 1}: reconstruction from the first k levels ==",
                     f"original: {decode_ids(bpe, ids[j].tolist())}"]
            for k, f in enumerate(f_hats):
                text = decode_ids(bpe, tok.decode(f)[j].argmax(-1).tolist())
                demo.append(f"level {k + 1}: {text}")
            demo.append("")
        _, canvases = sample(gen, tok, batch=3)
        for j in range(3):
            demo.append(f"== generated story {j + 1}: canvas decoded after each level ==")
            for k, c in enumerate(canvases):
                text = decode_ids(bpe, tok.decode(c)[j].argmax(-1).tolist())
                demo.append(f"level {k + 1}: {text}")
            demo.append("")
    (RESULTS_DIR / "phase2_level_by_level.txt").write_text("\n".join(demo), encoding="utf-8")

    # 3. sample quality
    def ncm_texts() -> list[str]:
        out = []
        for _ in range(N_SAMPLES // 128):
            _, canvases = sample(gen, tok, batch=128)
            out += [decode_ids(bpe, r.tolist()) for r in tok.decode(canvases[-1]).argmax(-1)]
        return out

    def ar_texts() -> list[str]:
        out = []
        for _ in range(N_SAMPLES // 128):
            out += [decode_ids(bpe, r.tolist()) for r in ar.generate(128)]
        return out

    with torch.no_grad(), torch.autocast(**amp):
        sets = {"real test stories": [decode_ids(bpe, r.tolist()) for r in test[:N_SAMPLES]],
                "toy NCM": ncm_texts(), "AR baseline": ar_texts()}
    report += ["", f"## Sample quality ({N_SAMPLES} samples, temperature 1.0)", "",
               "| source | GPT-2 perplexity (lower = more fluent) | distinct-2 |", "|---|---|---|"]
    for name, texts in sets.items():
        report.append(f"| {name} | {gpt2_perplexity(texts, dev):.1f} | {distinct_2(texts):.3f} |")
    with open(RESULTS_DIR / "phase2_samples.txt", "w", encoding="utf-8") as f:
        for name in ("toy NCM", "AR baseline"):
            f.write(f"===== {name} =====\n\n" + "\n\n".join(sets[name][:10]) + "\n\n")

    # 4. timing: 64 tokens per sample, bf16, generation only
    rows = []
    with torch.no_grad(), torch.autocast(**amp):
        for b in BATCH_SIZES:
            runs = {
                "toy NCM (6 passes)":
                    lambda b=b: tok.decode(sample(gen, tok, b)[1][-1]).argmax(-1),
                "AR + KV cache (64 passes)": lambda b=b: ar.generate(b),
                "AR no cache": lambda b=b: ar.generate(b, use_cache=False),
            }
            for name, fn in runs.items():
                sec = median_seconds(fn, dev)
                rows.append([name, b, round(sec * 1000, 2), round(b / sec, 1)])
    with open(RESULTS_DIR / "phase2_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "batch_size", "median_ms_per_batch", "samples_per_sec"])
        w.writerows(rows)
    report += ["", "## Generation time (64 tokens, median of 10 runs)", "",
               "| model | batch | ms per batch | samples/s |", "|---|---|---|---|"]
    report += [f"| {m} | {b} | {ms} | {sps} |" for m, b, ms, sps in rows]
    for b in BATCH_SIZES:
        ncm = next(r[2] for r in rows if r[0].startswith("toy NCM") and r[1] == b)
        kv = next(r[2] for r in rows if r[0].startswith("AR + KV") and r[1] == b)
        report.append(f"\nBatch {b}: toy NCM is {kv / ncm:.1f}x faster than AR with KV cache.")

    (RESULTS_DIR / "phase2_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("\n" + "\n".join(demo[:20]))


if __name__ == "__main__":
    main()
