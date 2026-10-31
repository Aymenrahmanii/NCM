"""Limit 2: throughput, latency and GPU memory from batch 1 up to out-of-memory.

Phase 4 measured batch 1, 8 and 32 only. This sweeps powers of two to find the exact batch
size where the AR model overtakes NCM, and the largest batch each model fits into 6 GB.
"""

from .common import (free_gpu, is_oom, load_model, load_task, model_label, peak_mb, reset_peak,
                     timed, write)

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
EXAMPLES = 128  # pushed through at every batch size (at least two batches each)


def sweep(model, label: str, task: str, inputs: list, batch_sizes=BATCH_SIZES,
          examples: int = EXAMPLES) -> list[dict]:
    rows = []
    for bs in batch_sizes:
        total = max(examples, 2 * bs)
        items = (inputs * (total // len(inputs) + 1))[:total]
        batches = [items[i : i + bs] for i in range(0, total, bs)]
        row = {"task": task, "model": label, "batch_size": bs, "examples": total}
        try:
            model.run(model.prepare(batches[0]))  # warm-up at this shape
            reset_peak()
            ms = [timed(model.run, model.prepare(b))[1] for b in batches]
        except Exception as e:
            if not is_oom(e):
                raise
            free_gpu()
            rows.append(row | {"oom": True})
            break  # larger batches will not fit either
        rows.append(row | {"oom": False, "examples_per_s": round(1000 * total / sum(ms), 2),
                           "ms_per_batch": round(sum(ms) / len(ms), 1),
                           "ms_per_example": round(sum(ms) / total, 2),
                           "peak_MB": peak_mb()})
    return rows


def crossover(rows: list[dict], a: str, b: str) -> int | None:
    """Smallest batch size from which model b's throughput beats model a's at every larger
    batch size too (a lone win at batch 1 followed by losses is not a crossover)."""
    tp = {(r["model"], int(r["batch_size"])): float(r["examples_per_s"])
          for r in rows if str(r.get("oom")) == "False"}  # rows from sweep() or a CSV
    shared = sorted(bs for m, bs in tp if m == a and (b, bs) in tp)
    found = None
    for bs in reversed(shared):
        if tp[(b, bs)] <= tp[(a, bs)]:
            break
        found = bs
    return found


def main(n: int = 128, precision: str = "fp16") -> None:
    rows = []
    for task in ("xsum", "wmt14"):
        ncm_inputs, text_inputs, _ = load_task(task, n)
        for kind in ("ncm", "ar"):
            model = load_model(kind, task, precision)
            rows += sweep(model, model_label(kind, task), task,
                          ncm_inputs if kind == "ncm" else text_inputs)
            del model
            free_gpu()
    write("batch_sweep", rows)
