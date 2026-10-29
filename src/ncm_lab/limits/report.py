"""Turn the limits CSVs into a Markdown summary, charts, and LaTeX tables for the paper.

    python -m ncm_lab.limits.report

Writes results/limits_summary.md, results/limits_*.png and paper/tables/limits_*.tex. Any
experiment that has not been run yet is simply left out.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

from .batch_sweep import crossover  # noqa: E402
from .common import RESULTS, read  # noqa: E402

PAPER_TABLES = Path("paper/tables")

# one fixed color + marker per kind of system (validated categorical slots 1-3); the marker
# and a direct label carry identity too, so the chart never relies on color alone
STYLE = {"ncm": ("#2a78d6", "o"), "ar": ("#eb6834", "s"), "llm": ("#1baf7a", "^")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def kind(name: str) -> str:
    n = name.lower()
    if "ncm" in n:
        return "ncm"
    return "llm" if ("llm" in n or "qwen" in n or "zero-shot" in n) else "ar"


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def line(ax, xs, ys, name: str) -> None:
    color, marker = STYLE[kind(name)]
    ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2, label=name)
    ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 0),
                va="center", fontsize=8, color=INK)


def log_axis(ax, values, base: int = 10) -> None:
    """Log scale with plain-number ticks at the data values (no 2^k or 10^k labels)."""
    ax.set_xscale("log", base=base)
    ticks = sorted({round(v) for v in values if v})
    if base == 10 and len(ticks) > 1:  # latencies: a few round numbers instead
        lo, hi = ticks[0], ticks[-1]
        ticks = [t for t in (50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000)
                 if lo / 1.5 <= t <= hi * 3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)


def save(fig, name: str, results: Path) -> Path:
    path = results / f"limits_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- tables --------------------------------------------------------------------------------
def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def tex_table(rows: list[dict], cols: list[str], caption: str, label: str) -> str:
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")  # noqa
    body = "\n".join(" & ".join(esc(r.get(c, "")) for c in cols) + r" \\" for r in rows)
    return (f"\\begin{{table}}[t]\n\\centering\\small\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\\begin{{tabular}}{{{'l' * 2 + 'r' * (len(cols) - 2)}}}\n"
            f"\\toprule\n{' & '.join(esc(c) for c in cols)} \\\\\n\\midrule\n{body}\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ---- one section per experiment: (title, rows, columns, caption, chart function) ----------
def chart_batch_sweep(rows, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task in zip(axes, ("xsum", "wmt14")):
        for model in dict.fromkeys(r["model"] for r in rows if r["task"] == task):
            pts = [(int(r["batch_size"]), num(r["examples_per_s"])) for r in rows
                   if r["task"] == task and r["model"] == model and r["oom"] == "False"]
            if pts:
                line(ax, *zip(*pts), model)
            oom = [int(r["batch_size"]) for r in rows
                   if r["task"] == task and r["model"] == model and r["oom"] == "True"]
            if pts and oom:
                ax.annotate(f"out of memory at batch {oom[0]}", pts[-1],
                            textcoords="offset points", xytext=(6, -12), fontsize=8,
                            color=MUTED)
        log_axis(ax, [int(r["batch_size"]) for r in rows if r["task"] == task], base=2)
        style_axes(ax, f"{task}: throughput vs batch size", "batch size", "examples / s")
    return save(fig, "batch_sweep", results)


def chart_output_length(rows, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task in zip(axes, ("xsum", "wmt14")):
        sub = [r for r in rows if r["task"] == task and r["batch_size"] == "1"]
        ar = [(int(r["output_tokens"]), num(r["ms_p50"])) for r in sub if r["model"] != "NCM"]
        if ar:
            line(ax, *zip(*ar), next(r["model"] for r in sub if r["model"] != "NCM"))
        ncm = [r for r in sub if r["model"] == "NCM"]
        if ncm:
            color, _ = STYLE["ncm"]
            ax.axhline(num(ncm[0]["ms_p50"]), color=color, linewidth=2, linestyle="--")
            ax.annotate("NCM: same cost for any output up to its canvas",
                        (ax.get_xlim()[1], num(ncm[0]["ms_p50"])), textcoords="offset points",
                        xytext=(-4, -12), ha="right", fontsize=8, color=INK)
        style_axes(ax, f"{task}: latency at batch 1 vs output length", "output tokens",
                   "ms (median)")
    return save(fig, "output_length", results)


def chart_input_length(rows, results):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for model in dict.fromkeys(r["model"] for r in rows):
        pts = [(num(r["input_words"]), num(r["bleu"])) for r in rows if r["model"] == model]
        line(ax, *zip(*pts), model)
    style_axes(ax, "WMT14 De→En: BLEU vs input length", "input words (mean)", "BLEU")
    return save(fig, "input_length", results)


def chart_llm_vs_ncm(rows, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (task, metric) in zip(axes, (("xsum", "rougeL"), ("wmt14", "bleu"))):
        for r in (r for r in rows if r["task"] == task):
            color, marker = STYLE[kind(r["system"])]
            x, y = num(r["latency_p50_ms"]), num(r[metric])
            ax.scatter([x], [y], color=color, marker=marker, s=60, zorder=3,
                       edgecolors="white", linewidths=2)
            ax.annotate(f"{r['system']}\n{r['params_M']}M params", (x, y),
                        textcoords="offset points", xytext=(8, -4), fontsize=8, color=INK)
        xs = [num(r["latency_p50_ms"]) for r in rows if r["task"] == task]
        if xs:
            log_axis(ax, xs)
            ax.set_xlim(min(xs) / 1.5, max(xs) * 3)
        ax.margins(y=0.15)
        style_axes(ax, f"{task}: quality vs latency (batch 1)", "median latency, ms (log)",
                   "ROUGE-L" if metric == "rougeL" else "BLEU")
    return save(fig, "llm_vs_ncm", results)


SECTIONS = [
    ("batch_sweep", "Throughput and memory vs batch size",
     ["task", "model", "batch_size", "examples_per_s", "ms_per_batch", "peak_MB", "oom"],
     "Throughput and peak GPU memory by batch size (RTX 3050 6\\,GB, fp16).", chart_batch_sweep),
    ("output_length_breakeven", "Output length where AR becomes slower than NCM",
     ["task", "batch_size", "ncm_ms", "ar_ms_fixed", "ar_ms_per_token", "breakeven_tokens",
      "ncm_canvas_tokens"],
     "Fitted AR latency (fixed + per-token) and the output length at which it exceeds NCM.",
     None),
    ("output_length", None, ["task", "model", "batch_size", "output_tokens", "ms_p50"],
     None, chart_output_length),
    ("input_length", "Translation quality vs input length",
     ["model", "sentences_per_input", "input_words", "input_tokens", "share_truncated", "bleu",
      "chrf", "length_ratio"],
     "WMT14 De$\\rightarrow$En with $k$ test sentences joined per input.", chart_input_length),
    ("faithfulness", "Invented numbers, names and repetition",
     ["task", "model", "n", "share_invented_number", "share_invented_name", "share_repetition",
      "share_any"],
     "Share of outputs with a number or name absent from the source, or a repetition.", None),
    ("robustness", "Quality under noisy input",
     ["task", "model", "perturbation", "rougeL", "bleu", "drop_vs_clean"],
     "Quality on perturbed inputs and the drop relative to clean input.", None),
    ("sudoku_stress", "Sudoku out of distribution",
     ["set", "solver", "n", "correct", "valid_solution", "full_grid_returned",
      "said_unsolvable", "ms_per_puzzle"],
     "Correct = valid solution, or for the unsolvable set, saying so.", None),
    ("llm_vs_ncm", "Specialist NCM vs a general 3B LLM",
     ["task", "system", "params_M", "rougeL", "bleu", "chrf", "output_words", "latency_p50_ms",
      "latency_p95_ms"],
     "NCM, the AR baseline and qwen2.5:3b (zero-shot, 4-bit) on the same examples.",
     chart_llm_vs_ncm),
    ("agent_reliability", "Agent reliability over sampled runs",
     ["task", "temperature", "runs", "right_tools", "runs_with_tool_error", "out_of_steps",
      "accuracy", "mean_llm_calls", "mean_total_ms"],
     "Agent runs per task: share calling the needed tools, tool errors, and accuracy.", None),
]


def main(results: Path = RESULTS, tables: Path = PAPER_TABLES) -> Path:
    tables.mkdir(parents=True, exist_ok=True)
    lines = ["# Limits of NCM: results", "",
             "Generated by `python -m ncm_lab.limits.report` from `results/limits_*.csv`.", ""]
    for name, title, cols, caption, chart in SECTIONS:
        rows = read(name, results)
        if not rows:
            continue
        cols = [c for c in cols if any(r.get(c) not in (None, "") for r in rows)]
        if title:
            lines += [f"## {title}", "", md_table(rows, cols), ""]
            (tables / f"limits_{name}.tex").write_text(
                tex_table(rows, cols, caption, f"tab:{name}"), encoding="utf-8")
        if name == "batch_sweep":
            for task in ("xsum", "wmt14"):
                models = list(dict.fromkeys(r["model"] for r in rows if r["task"] == task))
                if len(models) == 2:
                    bs = crossover([r for r in rows if r["task"] == task], *models)
                    lines += [f"- {task}: {models[1]} overtakes {models[0]} from batch size "
                              f"{bs}." if bs else f"- {task}: no crossover in the sweep.", ""]
        if chart:
            path = chart(rows, results)
            lines += [f"![{name}]({path.name})", ""]
    out = results / "limits_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
