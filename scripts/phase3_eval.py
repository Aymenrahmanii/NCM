"""Phase 3: run the official checkpoints through our wrapper and check scores and memory.

    python scripts/phase3_eval.py            # 50 examples per task
    python scripts/phase3_eval.py --n 200

Also runs the official evaluate.py on the same examples as a cross-check.
"""

import argparse
import csv
import gc
import pickle
import subprocess
import sys
import time
from pathlib import Path

import torch

from ncm_lab.models.ncm import NCM_ROOT, NCMSummarizer, NCMTranslator

RESULTS = Path("results")
PAPER = {"xsum": "ROUGE-L 28.0 (full test set)", "wmt14": "BLEU 22.6 (full test set)"}


def rouge_l(hyps: list[str], refs: list[str]) -> dict[str, float]:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [sc.score(r, h) for h, r in zip(hyps, refs)]
    return {k: 100 * sum(s[k].fmeasure for s in scores) / len(scores)
            for k in ("rouge1", "rouge2", "rougeL")}


def bleu(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(hyps, [refs]).score


def run(cls, inputs, refs, precision: str, metric) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = cls(precision=precision)
    weights_mb = torch.cuda.memory_allocated() / 2**20
    model.generate(inputs[:4])  # warm-up
    torch.cuda.synchronize()
    start = time.perf_counter()
    hyps = model.generate(inputs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / 2**20
    row = {"task": model.task, "precision": precision, "n": len(inputs),
           "params_M": round(model.n_params / 1e6, 1), "weights_MB": round(weights_mb),
           "peak_MB": round(peak_mb), "seconds": round(seconds, 2),
           "ms_per_example": round(1000 * seconds / len(inputs), 1)}
    row.update({k: round(v, 2) for k, v in metric(hyps, refs).items()})
    del model
    gc.collect()
    return row, hyps


def official(task: str, n: int) -> str:
    """Run the official evaluate.py on the first n examples; return its score lines."""
    sub = Path(*task.split("-")) if task.startswith("sudoku") else Path(task)  # sudoku/easy
    out = subprocess.run(
        [sys.executable, "evaluate.py", "--config", f"config/eval/{task}.yaml",
         "--checkpoint-dir", str(Path("artifacts", sub).resolve()),
         "--data-dir", str(Path("data", sub).resolve()),
         "--out", str(Path("artifacts/official_eval", task).resolve()),
         "--gpus", "0", "--n", str(n)],
        cwd=NCM_ROOT, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip-official", action="store_true")
    args = ap.parse_args()
    n = args.n

    xsum_ids = pickle.load(open("data/xsum/passages.pickle", "rb"))[:n]
    xsum_refs = pickle.load(open("data/xsum/reference_summaries.pickle", "rb"))[:n]
    wmt_src = pickle.load(open("data/wmt14/sources.pickle", "rb"))[:n]
    wmt_refs = pickle.load(open("data/wmt14/reference_translations.pickle", "rb"))[:n]

    rows, examples = [], []
    for precision in ("fp32", "fp16", "bf16"):
        row, hyps = run(NCMSummarizer, xsum_ids, xsum_refs, precision, rouge_l)
        rows.append(row)
        print(row, flush=True)
        if precision == "fp32":
            examples += [("xsum", h, r) for h, r in zip(hyps[:5], xsum_refs[:5])]
        row, hyps = run(NCMTranslator, wmt_src, wmt_refs, precision,
                        lambda h, r: {"bleu": bleu(h, r)})
        rows.append(row)
        print(row, flush=True)
        if precision == "fp32":
            examples += [("wmt14", h, r) for h, r in zip(hyps[:5], wmt_refs[:5])]

    # raw-text path: decode the eval ids to a string and let the wrapper re-tokenize it.
    # Each eval item is 560 ids: article + <pad> (50257) up to 512, then the 48-token GOLD
    # summary. Keep only the article part, or the model would be shown the answer.
    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    raw = [gpt2.decode([t for t in ids[:512] if t < 50256]) for ids in xsum_ids]
    row, _ = run(NCMSummarizer, raw, xsum_refs, "fp32", rouge_l)
    row["task"] = "xsum (raw text input)"
    rows.append(row)
    print(row, flush=True)

    RESULTS.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                  if k in rows[0] else 99)
    with open(RESULTS / "phase3_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = [f"# Phase 3: official NCM checkpoints on an RTX 3050 (first {n} test examples)", "",
             "| task | precision | params | weights MB | peak MB | ms/example | score |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        score = (f"R-1 {r['rouge1']} / R-2 {r['rouge2']} / R-L {r['rougeL']}" if "rougeL" in r
                 else f"BLEU {r['bleu']}")
        lines.append(f"| {r['task']} | {r['precision']} | {r['params_M']}M | {r['weights_MB']} "
                     f"| {r['peak_MB']} | {r['ms_per_example']} | {score} |")
    lines += ["", f"Paper: XSum {PAPER['xsum']}, WMT14 {PAPER['wmt14']}.", ""]

    if not args.skip_official:
        for task in ("xsum", "wmt14", "sudoku-easy", "sudoku-medium", "sudoku-hard"):
            task_n = 200 if task.startswith("sudoku") else n
            print(f"official {task} ...", flush=True)
            report = official(task, task_n)
            lines += [f"## Official evaluate.py: {task} (n={task_n})", "", "```",
                      report[-1500:], "```", ""]

    lines += ["## Examples (fp32)", ""]
    for task, h, r in examples:
        lines += [f"- **{task}** output: {h}", f"  reference: {r}"]
    (RESULTS / "phase3_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
