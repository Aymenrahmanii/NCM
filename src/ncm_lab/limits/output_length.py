"""Limit 3: how long must outputs be before NCM's fixed step count beats AR decoding?

NCM's latency does not depend on how much it writes (a fixed number of level passes over a
fixed-size canvas); an AR model pays one decoder step per token. The task checkpoints only
write ~20 words, which is AR's best case. Here the AR baseline is forced to write exactly L
tokens (min_new_tokens = max_new_tokens = L) and its latency is compared with NCM's constant
cost. This measures speed only: forced long AR outputs are not meaningful text, and NCM
cannot write longer outputs than its canvas at all.
"""

import statistics

from .common import free_gpu, load_model, load_task, model_label, pct, timed, write

LENGTHS = (16, 32, 64, 128, 256, 400)
BATCH_SIZES = (1, 8)
REPEATS = 20


def ncm_cost(model, inputs: list, batch_sizes=BATCH_SIZES, repeats: int = REPEATS):
    rows = []
    for bs in batch_sizes:
        x = model.prepare(inputs[:bs])
        tokens = model.run(x)  # warm-up; also tells us the canvas size
        ms = [timed(model.run, x)[1] for _ in range(repeats)]
        canvas = int(tokens.shape[1]) if hasattr(tokens, "shape") else None
        rows.append({"model": "NCM", "batch_size": bs, "output_tokens": canvas,
                     "ms_p50": round(statistics.median(ms), 1), "ms_p95": pct(ms, 0.95)})
    return rows


def ar_cost(model, label: str, inputs: list[str], lengths=LENGTHS, batch_sizes=BATCH_SIZES,
            repeats: int = REPEATS):
    rows = []
    for bs in batch_sizes:
        x = model.prepare(inputs[:bs])
        for length in lengths:
            run = lambda b, n=length: model.model.generate(  # noqa: E731
                **b, min_new_tokens=n, max_new_tokens=n, do_sample=False, num_beams=1)
            run(x)  # warm-up
            ms = [timed(run, x)[1] for _ in range(repeats)]
            rows.append({"model": label, "batch_size": bs, "output_tokens": length,
                         "ms_p50": round(statistics.median(ms), 1), "ms_p95": pct(ms, 0.95)})
    return rows


def fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares ms = a + b * tokens."""
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def breakeven(rows: list[dict], task: str) -> list[dict]:
    """Per batch size: AR ms/token and the output length where AR becomes slower than NCM."""
    out = []
    for bs in sorted({int(r["batch_size"]) for r in rows}):
        ncm = [r for r in rows if r["model"] == "NCM" and int(r["batch_size"]) == bs]
        ar = [(float(r["output_tokens"]), float(r["ms_p50"])) for r in rows
              if r["model"] != "NCM" and int(r["batch_size"]) == bs]
        if not ncm or len(ar) < 2:
            continue
        a, b = fit_line(ar)
        ncm_ms = float(ncm[0]["ms_p50"])
        out.append({"task": task, "batch_size": bs, "ncm_ms": ncm_ms,
                    "ar_ms_fixed": round(a, 1), "ar_ms_per_token": round(b, 3),
                    "breakeven_tokens": round((ncm_ms - a) / b) if b > 0 else None,
                    "ncm_canvas_tokens": ncm[0]["output_tokens"]})
    return out


def main(n: int = 32, precision: str = "fp16") -> None:
    rows, summary = [], []
    for task in ("xsum", "wmt14"):
        ncm_inputs, text_inputs, _ = load_task(task, n)
        model = load_model("ncm", task, precision)
        task_rows = ncm_cost(model, ncm_inputs)
        del model
        free_gpu()
        model = load_model("ar", task, precision)
        task_rows += ar_cost(model, model_label("ar", task), text_inputs)
        del model
        free_gpu()
        rows += [r | {"task": task} for r in task_rows]
        summary += breakeven(task_rows, task)
    write("output_length", rows)
    write("output_length_breakeven", summary)
