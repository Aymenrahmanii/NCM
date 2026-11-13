"""Latency under the research plan's protocol: interleaved models, warm-up, many timed calls.

    NCM_ROOT=third_party/ncm-pr5-pr4 python -m ncm_lab.bench.interleaved --task wmt14

For each batch size: 20 warm-up calls per model (compiled models: one per distinct input
batch, so shape-specialized recompiles stay out of the timing), then 200 rounds. Every
round runs each model once on the same input batch, in an order that rotates from round to
round, so clock and
temperature drift hit all models alike. Each call is timed with torch.cuda.synchronize()
around it; inputs are prepared before the clock starts.

Models (--models): the NCM variants of ncm_lab.bench.run.NCM_VARIANTS ("ncm", "ncm-kv",
"ncm-half", "ncm-kv-half"; "kv" needs NCM_ROOT pointing at a checkout with upstream PR #5,
"half" stores the weights in --precision, "ncm-kv-half-compiled" also compiles the cached
passes with CUDA graphs), "ar" (the task's AR baseline) and "ar-compiled"
(the same with a static cache and a compiled decoding step). NCM variants that differ only in
the cache share one copy of the weights.

    python -m ncm_lab.bench.interleaved --task wmt14 --device cpu --threads 4 \
        --precision fp32 --batch-sizes 1 --calls 50 --warmup 5 --tag cpu4

Writes results/phase8_interleaved_<task>[_<tag>].csv (median / p95 latency and throughput
with bootstrap 95% CIs), ..._ratios.csv (paired ratios of median latency for every model
pair) and ..._calls.jsonl (every timed call, plus GPU temperature and SM clock every 25
rounds on CUDA).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import time
from pathlib import Path

import torch

from ncm_lab.bench.run import NCM_VARIANTS, load_task, make_model
from ncm_lab.stats import bootstrap_quantile, paired_ratio_of_medians

RESULTS = Path("results")
MODELS = ("ncm", "ncm-kv", "ar")


def gpu_state() -> dict:
    try:
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
               "--format=csv,noheader,nounits"]
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip().split(", ")
        return {"temp_c": int(out[0]), "sm_mhz": int(out[1])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"temp_c": None, "sm_mhz": None}


def _sync() -> None:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def timed(model, prepared) -> float:
    _sync()
    start = time.perf_counter()
    model.run(prepared)
    _sync()
    return (time.perf_counter() - start) * 1000


def batches(items: list, bs: int, k: int) -> list[list]:
    """k batches of size bs, wrapping around the data set."""
    n = len(items)
    return [[items[(i * bs + j) % n] for j in range(bs)] for i in range(k)]


def run_batch_size(models: dict, inputs: dict, bs: int, calls: int, warmup: int,
                   log: list[dict]) -> dict[str, list[float]]:
    k = min(calls, max(1, min(len(x) for x in inputs.values()) // bs))  # distinct batches
    prepared = {
        name: [models[name].prepare(b) for b in batches(inputs[name], bs, k)] for name in models
    }
    for name, model in models.items():
        # Compiled graphs are specialized per input shape: warm up on every distinct batch so
        # no compilation happens inside the timed rounds.
        n_warm = max(warmup, k) if getattr(model, "compiled", False) else warmup
        for i in range(n_warm):
            model.run(prepared[name][i % k])
    times: dict[str, list[float]] = {name: [] for name in models}
    names = list(models)
    for r in range(calls):
        if r % 25 == 0 and torch.cuda.is_initialized():
            log.append({"batch_size": bs, "round": r, **gpu_state()})
        shift = r % len(names)
        for name in names[shift:] + names[:shift]:
            ms = timed(models[name], prepared[name][r % k])
            times[name].append(ms)
            log.append({"batch_size": bs, "round": r, "model": name, "ms": round(ms, 3)})
    if torch.cuda.is_initialized():
        log.append({"batch_size": bs, "round": calls, **gpu_state()})
    return times


class _KVView:
    """A loaded NCM model with the cache set for each call (no second copy of the weights)."""

    def __init__(self, ncm, on: bool):
        self.ncm, self.on = ncm, on
        self.compiled = getattr(ncm, "compiled", False)

    def prepare(self, items):
        return self.ncm.prepare(items)

    def run(self, x):
        self.ncm.kv_cache = self.on
        return self.ncm.run(x)


def write_results(stem: Path, rows: list[dict], ratio_rows: list[dict],
                  log: list[dict]) -> None:
    stem.parent.mkdir(exist_ok=True)
    for path, data in ((stem.with_suffix(".csv"), rows),
                       (Path(f"{stem}_ratios.csv"), ratio_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    with open(f"{stem}_calls.jsonl", "w", encoding="utf-8") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task", choices=["xsum", "wmt14"], required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--n", type=int, default=0, help="test examples to draw inputs from (0 = all)")
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    choices=[*NCM_VARIANTS, "ar", "ar-compiled"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    ncm_inputs, ar_inputs, _ = load_task(args.task, args.n or 10**9)
    torch.manual_seed(0)
    models, inputs, shared = {}, {}, {}
    for name in args.models:
        if name in NCM_VARIANTS:
            kv, cast, comp = NCM_VARIANTS[name]
            key = (cast, comp)  # a compiled model is never shared with an eager variant
            if key not in shared:
                shared[key] = make_model(name, args.task, args.precision, args.device)
            models[name], inputs[name] = _KVView(shared[key], kv), ncm_inputs
        else:
            models[name] = make_model(name, args.task, args.precision, args.device)
            inputs[name] = ar_inputs

    stem = RESULTS / f"phase8_interleaved_{args.task}{'_' + args.tag if args.tag else ''}"
    on_gpu = args.device == "cuda"
    device_name = torch.cuda.get_device_name() if on_gpu else f"cpu x{torch.get_num_threads()}"
    rows, ratio_rows, log = [], [], [{"start": gpu_state() if on_gpu else {},
                                      "torch": torch.__version__, "device": device_name}]
    for bs in args.batch_sizes:
        print(f"[interleaved] {args.task} batch {bs}: {args.calls} rounds", flush=True)
        times = run_batch_size(models, inputs, bs, args.calls, args.warmup, log)
        for name, t in times.items():
            med, (mlo, mhi) = bootstrap_quantile(t, 0.5)
            p95, (plo, phi) = bootstrap_quantile(t, 0.95)
            rows.append({
                "task": args.task, "model": name, "precision": args.precision,
                "device": device_name, "batch_size": bs, "calls": len(t),
                "median_ms": round(med, 1), "median_lo": round(mlo, 1), "median_hi": round(mhi, 1),
                "p95_ms": round(p95, 1), "p95_lo": round(plo, 1), "p95_hi": round(phi, 1),
                "ex_per_s": round(1000 * bs / med, 2),
                "ex_per_s_lo": round(1000 * bs / mhi, 2), "ex_per_s_hi": round(1000 * bs / mlo, 2),
            })
        for a, b in itertools.combinations(args.models, 2):
            r, (lo, hi) = paired_ratio_of_medians(times[a], times[b])
            # ratio > 1: model_a is slower than model_b
            ratio_rows.append({"task": args.task, "batch_size": bs, "model_a": a, "model_b": b,
                               "median_latency_ratio_a_over_b": round(r, 3),
                               "lo": round(lo, 3), "hi": round(hi, 3),
                               "differs": not lo <= 1.0 <= hi})
        for row in rows[-len(times):]:
            print(row, flush=True)
        # Rewritten after every batch size so a crash keeps the finished ones.
        write_results(stem, rows, ratio_rows, log)

    for r in ratio_rows:
        print(r)
    print(f"wrote {stem}.csv")


if __name__ == "__main__":
    main()
