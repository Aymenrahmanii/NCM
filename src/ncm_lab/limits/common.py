"""Shared helpers: timing, GPU memory, model loading, metrics and result files.

Everything here also works on a CPU-only machine (the tests run every experiment with tiny
stand-in models), so CUDA calls are skipped when there is no GPU.
"""

import csv
import gc
import json
import random
import statistics
import time
from pathlib import Path

import torch

RESULTS = Path("results")


def cuda() -> bool:
    return torch.cuda.is_available()


def sync() -> None:
    if cuda():
        torch.cuda.synchronize()


def timed(fn, *args) -> tuple[object, float]:
    """(result, milliseconds), with the GPU synchronized around the clock."""
    sync()
    start = time.perf_counter()
    out = fn(*args)
    sync()
    return out, (time.perf_counter() - start) * 1000


def reset_peak() -> None:
    if cuda():
        torch.cuda.reset_peak_memory_stats()


def peak_mb() -> int | None:
    return round(torch.cuda.max_memory_allocated() / 2**20) if cuda() else None


def free_gpu() -> None:
    """Call after `del model` so the next model gets the memory back."""
    gc.collect()
    if cuda():
        torch.cuda.empty_cache()


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 1)


def mean(xs) -> float:
    xs = list(xs)
    return round(statistics.fmean(xs), 3) if xs else 0.0


# ---- models --------------------------------------------------------------------------------
def load_model(kind: str, task: str, precision: str = "fp16", **kw):
    """kind: "ncm" (official checkpoint) or "ar" (the Phase 4 same-size baseline)."""
    if kind == "ncm":
        from ncm_lab.models.ncm import NCMSummarizer, NCMTranslator

        return (NCMSummarizer if task == "xsum" else NCMTranslator)(precision=precision, **kw)
    if kind == "ar":
        from ncm_lab.models.baselines import HFSeq2Seq

        return HFSeq2Seq(task, precision=precision, **kw)
    raise ValueError(kind)


def model_label(kind: str, task: str) -> str:
    from ncm_lab.models.baselines import BASELINES

    return f"NCM ({task})" if kind == "ncm" else BASELINES[task].split("/")[-1]


# ---- data ----------------------------------------------------------------------------------
def load_task(task: str, n: int) -> tuple[list, list[str], list[str]]:
    """(NCM inputs, text inputs, references); see ncm_lab.bench.run.load_task."""
    from ncm_lab.bench.run import load_task as bench_load

    return bench_load(task, n)


# ---- metrics -------------------------------------------------------------------------------
def quality(task: str, hyps: list[str], refs: list[str]) -> dict[str, float]:
    """ROUGE-1/2/L for xsum, BLEU + chrF for wmt14 (sacreBLEU defaults)."""
    if task == "xsum":
        from ncm_lab.bench.run import score

        return score(task, hyps, refs)
    import sacrebleu

    return {"bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}


def main_metric(task: str) -> str:
    return "rougeL" if task == "xsum" else "bleu"


def words(texts: list[str]) -> float:
    return round(statistics.fmean(len(t.split()) for t in texts), 1) if texts else 0.0


# ---- result files --------------------------------------------------------------------------
def write(name: str, rows: list[dict], details: list[dict] | None = None,
          results: Path = RESULTS) -> Path:
    """results/limits_<name>.csv (one row per condition) + optional _details.jsonl."""
    results.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))  # union, first-seen order
    path = results / f"limits_{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if details:
        with open(results / f"limits_{name}_details.jsonl", "w", encoding="utf-8") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    for r in rows:
        print("  ", {k: v for k, v in r.items() if not isinstance(v, list)})
    return path


def read(name: str, results: Path = RESULTS) -> list[dict] | None:
    path = results / f"limits_{name}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
