"""Charts for Phase 4: quality vs latency, and throughput vs batch size.

    python -m ncm_lab.bench.plot
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
QUALITY = {"xsum": ("rougeL", "ROUGE-L"), "wmt14": ("bleu", "BLEU")}
COLORS = {"ncm": "#2a6fdb", "ar": "#d9822b"}
LABELS = {"ncm": "NCM", "ar": "Autoregressive"}


def main() -> None:
    for path in sorted(RESULTS.glob("phase4_*_*.csv")):
        rows = list(csv.DictReader(open(path)))
        if not rows or "latency_p50_ms" not in rows[0]:
            continue
        task, precision = rows[0]["task"], rows[0]["precision"]
        key, label = QUALITY[task]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        for r in rows:
            c = COLORS.get(r["model"], "gray")
            ax1.errorbar(float(r["latency_p50_ms"]), float(r[key]),
                         xerr=[[0], [float(r["latency_p95_ms"]) - float(r["latency_p50_ms"])]],
                         fmt="o", color=c, capsize=4, markersize=9)
            ax1.annotate(f"{LABELS.get(r['model'], r['model'])}\n{r['params_M']}M",
                         (float(r["latency_p50_ms"]), float(r[key])),
                         textcoords="offset points", xytext=(8, 4), fontsize=9)
            bss = [1, 8, 32]
            ax2.plot(bss, [float(r[f"throughput_bs{b}"]) for b in bss], marker="o", color=c,
                     label=LABELS.get(r["model"], r["model"]))

        ax1.set_xlabel("latency per request, batch 1 (ms, median; bar to p95)")
        ax1.set_ylabel(label)
        ax1.set_title(f"{task}: quality vs latency ({precision})")
        ax1.grid(alpha=0.3)
        lo, hi = ax1.get_xlim()
        ax1.set_xlim(0, hi * 1.3)
        ax1.margins(y=0.25)  # room for the labels above the points
        ax2.set_xscale("log", base=2)
        ax2.set_xticks([1, 8, 32], ["1", "8", "32"])
        ax2.set_xlabel("batch size")
        ax2.set_ylabel("examples / second")
        ax2.set_title(f"{task}: throughput")
        ax2.grid(alpha=0.3)
        ax2.legend()
        fig.tight_layout()
        out = RESULTS / f"phase4_{task}_{precision}.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
