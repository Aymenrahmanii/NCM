"""Latency under the research plan's protocol: interleaved models, warm-up, many timed calls.

    NCM_ROOT=third_party/ncm-pr5-pr4 python -m ncm_lab.bench.interleaved --task wmt14

For each batch size: 20 warm-up calls per model, then 200 rounds. Every round runs each model
once on the same input batch, in an order that rotates from round to round, so clock and
temperature drift hit all models alike. Each call is timed with torch.cuda.synchronize()
around it; inputs are prepared before the clock starts.

Models: "ncm" (official uncached generator), "ncm-kv" (same weights with the KV cache; needs
NCM_ROOT pointing at a checkout with upstream PR #5) and "ar" (the task's AR baseline).

Writes results/phase8_interleaved_<task>.csv (median / p95 latency and throughput with
bootstrap 95% CIs), ..._ratios.csv (paired ratios of median latency) and ..._calls.jsonl
(every timed call, plus GPU temperature and SM clock every 25 rounds).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from ncm_lab.bench.run import load_task, make_model
from ncm_lab.stats import bootstrap_quantile, paired_ratio_of_medians

RESULTS = Path("results")
MODELS = ("ncm", "ncm-kv", "ar")
RATIOS = (("ncm", "ar"), ("ncm-kv", "ar"), ("ncm", "ncm-kv"))


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


def timed(model, prepared) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.run(prepared)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def batches(items: list, bs: int, k: int) -> list[list]:
    """k batches of size bs, wrapping around the data set."""
    n = len(items)
    return [[items[(i * bs + j) % n] for j in range(bs)] for i in range(k)]


def run_batch_size(models: dict, inputs: dict, bs: int, calls: int, warmup: int,
                   log: list[dict]) -> dict[str, list[float]]:
    k = min(calls, max(1, len(inputs["ncm"]) // bs))  # distinct batches, cycled
    prepared = {
        name: [models[name].prepare(b) for b in batches(inputs[name], bs, k)] for name in models
    }
    for name, model in models.items():
        for i in range(warmup):
            model.run(prepared[name][i % k])
    times: dict[str, list[float]] = {name: [] for name in models}
    names = list(models)
    for r in range(calls):
        if r % 25 == 0:
            log.append({"batch_size": bs, "round": r, **gpu_state()})
        shift = r % len(names)
        for name in names[shift:] + names[:shift]:
            ms = timed(models[name], prepared[name][r % k])
            times[name].append(ms)
            log.append({"batch_size": bs, "round": r, "model": name, "ms": round(ms, 3)})
    log.append({"batch_size": bs, "round": calls, **gpu_state()})
    return times


class _KVView:
    """The loaded NCM model with the cache switched on for each call (no second copy)."""

    def __init__(self, ncm, on: bool):
        self.ncm, self.on = ncm, on

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
    args = ap.parse_args()

    ncm_inputs, ar_inputs, _ = load_task(args.task, args.n or 10**9)
    torch.manual_seed(0)
    ncm = make_model("ncm", args.task, args.precision)
    models = {"ncm": _KVView(ncm, False), "ncm-kv": _KVView(ncm, True),
              "ar": make_model("ar", args.task, args.precision)}
    inputs = {"ncm": ncm_inputs, "ncm-kv": ncm_inputs, "ar": ar_inputs}

    stem = RESULTS / f"phase8_interleaved_{args.task}"
    rows, ratio_rows, log = [], [], [{"start": gpu_state(), "torch": torch.__version__,
                                      "device": torch.cuda.get_device_name()}]
    for bs in args.batch_sizes:
        print(f"[interleaved] {args.task} batch {bs}: {args.calls} rounds", flush=True)
        times = run_batch_size(models, inputs, bs, args.calls, args.warmup, log)
        for name, t in times.items():
            med, (mlo, mhi) = bootstrap_quantile(t, 0.5)
            p95, (plo, phi) = bootstrap_quantile(t, 0.95)
            rows.append({
                "task": args.task, "model": name, "precision": args.precision,
                "batch_size": bs, "calls": len(t),
                "median_ms": round(med, 1), "median_lo": round(mlo, 1), "median_hi": round(mhi, 1),
                "p95_ms": round(p95, 1), "p95_lo": round(plo, 1), "p95_hi": round(phi, 1),
                "ex_per_s": round(1000 * bs / med, 2),
                "ex_per_s_lo": round(1000 * bs / mhi, 2), "ex_per_s_hi": round(1000 * bs / mlo, 2),
            })
        for a, b in RATIOS:
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
