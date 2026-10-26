"""Phase 4 benchmark harness: speed is always reported next to quality.

    python -m ncm_lab.bench.run --task xsum --models ncm ar --n 500
    python -m ncm_lab.bench.run --task wmt14 --models ncm ar --n 500 --precision fp16

Fairness rules: same GPU, precision, examples and seed; warm-up before timing;
torch.cuda.synchronize() around the clock; only generation (`run`) is timed, not tokenization
or loading; the AR baseline keeps its KV cache.
"""

import argparse
import csv
import gc
import json
import pickle
import statistics
import time
from pathlib import Path

import torch

RESULTS = Path("results")
LATENCY_N = 100          # single requests timed at batch size 1
THROUGHPUT_N = 256       # examples pushed through at batch sizes 8 and 32
BATCH_SIZES = (8, 32)


def load_task(task: str, n: int):
    """Inputs for NCM, inputs for the AR model (text), references."""
    if task == "xsum":
        from transformers import AutoTokenizer

        ids = pickle.load(open("data/xsum/passages.pickle", "rb"))[:n]
        refs = pickle.load(open("data/xsum/reference_summaries.pickle", "rb"))[:n]
        gpt2 = AutoTokenizer.from_pretrained("gpt2")
        # the AR model reads the same 512-token article prefix NCM sees, as text.
        # ids[512:] holds the gold summary and 50257 is padding: both must stay out.
        texts = [gpt2.decode([t for t in x[:512] if t < 50256]) for x in ids]
        return ids, texts, refs
    if task == "wmt14":
        src = pickle.load(open("data/wmt14/sources.pickle", "rb"))[:n]
        refs = pickle.load(open("data/wmt14/reference_translations.pickle", "rb"))[:n]
        return src, src, refs
    raise ValueError(task)


def score(task: str, hyps: list[str], refs: list[str]) -> dict[str, float]:
    if task == "xsum":
        from rouge_score import rouge_scorer

        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        s = [sc.score(r, h) for h, r in zip(hyps, refs)]
        return {k: round(100 * statistics.fmean(x[k].fmeasure for x in s), 2)
                for k in ("rouge1", "rouge2", "rougeL")}
    import sacrebleu

    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)}


def make_model(name: str, task: str, precision: str):
    if name == "ncm":
        from ncm_lab.models.ncm import NCMSummarizer, NCMTranslator

        cls = NCMSummarizer if task == "xsum" else NCMTranslator
        return cls(precision=precision)
    if name == "ar":
        from ncm_lab.models.baselines import HFSeq2Seq

        return HFSeq2Seq(task, precision=precision)
    raise ValueError(name)


def timed_run(model, prepared) -> tuple[object, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.run(prepared)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def bench_model(name: str, task: str, precision: str, inputs: list, refs: list[str]) -> dict:
    torch.manual_seed(0)
    model = make_model(name, task, precision)
    row = {"task": task, "model": name,
           "model_id": getattr(model, "model_id", f"hazyresearch/ncm-checkpoints/{task}"),
           "precision": precision, "n": len(inputs),
           "params_M": round(getattr(model, "n_params_total", model.n_params) / 1e6, 1)}

    # quality on all n examples, batch 32
    hyps = []
    for i in range(0, len(inputs), 32):
        out = model.run(model.prepare(inputs[i : i + 32]))
        hyps += model.finish(out)
    row.update(score(task, hyps, refs))
    row["avg_output_words"] = round(statistics.fmean(len(h.split()) for h in hyps), 1)

    # latency: one request at a time
    singles = [model.prepare([x]) for x in inputs[:LATENCY_N]]
    for p in singles[:3]:
        model.run(p)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    lat = sorted(timed_run(model, p)[1] * 1000 for p in singles)
    row["latency_p50_ms"] = round(statistics.median(lat), 1)
    row["latency_p95_ms"] = round(lat[int(0.95 * (len(lat) - 1))], 1)
    row["throughput_bs1"] = round(1000 / statistics.fmean(lat), 2)
    row["peak_MB_bs1"] = round(torch.cuda.max_memory_allocated() / 2**20)

    # throughput at larger batches
    for bs in BATCH_SIZES:
        batches = [model.prepare(inputs[i : i + bs]) for i in range(0, THROUGHPUT_N, bs)]
        model.run(batches[0])  # warm-up
        torch.cuda.reset_peak_memory_stats()
        total = sum(timed_run(model, b)[1] for b in batches)
        row[f"throughput_bs{bs}"] = round(THROUGHPUT_N / total, 2)
        row[f"peak_MB_bs{bs}"] = round(torch.cuda.max_memory_allocated() / 2**20)

    out_path = RESULTS / f"phase4_outputs_{task}_{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for h, r in zip(hyps, refs):
            f.write(json.dumps({"hyp": h, "ref": r}, ensure_ascii=False) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xsum", "wmt14"], required=True)
    ap.add_argument("--models", nargs="+", default=["ncm", "ar"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()

    ncm_inputs, ar_inputs, refs = load_task(args.task, args.n)
    rows = []
    for name in args.models:
        print(f"[bench] {args.task} / {name} / {args.precision}", flush=True)
        row = bench_model(name, args.task, args.precision,
                          ncm_inputs if name == "ncm" else ar_inputs, refs)
        print(row, flush=True)
        rows.append(row)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"phase4_{args.task}_{args.precision}.csv"
    if path.exists():  # keep rows of models not re-run this time
        done = {r["model"] for r in rows}
        old = [r for r in csv.DictReader(open(path)) if r["model"] not in done]
        rows = old + rows
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
