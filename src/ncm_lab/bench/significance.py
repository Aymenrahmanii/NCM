"""Confidence intervals for results that are already on disk (no GPU needed).

    python -m ncm_lab.bench.significance

Phase 4: paired bootstrap (10,000 resamples) of NCM vs AR on the full test sets, from
results/phase4_outputs_*.jsonl. Phase 5 RAG: Wilson intervals per variant and exact McNemar
tests between NCM and AR variants, from results/phase5_rag_*_details.jsonl.
Writes results/significance.csv and results/significance.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ncm_lab.stats import (
    N_RESAMPLES,
    PairedResult,
    mcnemar_exact,
    paired_bootstrap_bleu,
    paired_bootstrap_mean,
    wilson_interval,
)

RESULTS = Path("results")
RAG_PAIRS = {  # experiment -> (AR variant, NCM variant)
    "phase5_rag_multilingual": (
        "translated by opus-mt-de-en (AR)", "translated by NCM (WMT14)",
    ),
    "phase5_rag_compression": (
        "AR-compressed (BART-base XSum)", "NCM-compressed (XSum summarizer)",
    ),
}


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _row(task: str, metric: str, r: PairedResult, scale: float = 1.0) -> dict:
    def f(x: float) -> float:
        return round(x * scale, 2)

    return {
        "task": task, "metric": metric, "n": r.n,
        "ar": f(r.a), "ar_lo": f(r.a_ci[0]), "ar_hi": f(r.a_ci[1]),
        "ncm": f(r.b), "ncm_lo": f(r.b_ci[0]), "ncm_hi": f(r.b_ci[1]),
        "ncm_minus_ar": f(r.delta), "delta_lo": f(r.delta_ci[0]), "delta_hi": f(r.delta_ci[1]),
        "significant": r.significant,
    }


def phase4_rows(n_resamples: int) -> list[dict]:
    rows = []
    ar = _jsonl(RESULTS / "phase4_outputs_wmt14_ar.jsonl")
    ncm = _jsonl(RESULTS / "phase4_outputs_wmt14_ncm.jsonl")
    refs = [x["ref"] for x in ar]
    r = paired_bootstrap_bleu([x["hyp"] for x in ar], [x["hyp"] for x in ncm], refs, n_resamples)
    rows.append(_row("wmt14", "BLEU", r))

    from rouge_score import rouge_scorer

    ar = _jsonl(RESULTS / "phase4_outputs_xsum_ar.jsonl")
    ncm = _jsonl(RESULTS / "phase4_outputs_xsum_ncm.jsonl")
    keys = ("rouge1", "rouge2", "rougeL")
    sc = rouge_scorer.RougeScorer(list(keys), use_stemmer=True)
    s_ar = [sc.score(x["ref"], x["hyp"]) for x in ar]
    s_ncm = [sc.score(x["ref"], x["hyp"]) for x in ncm]
    for k in keys:
        r = paired_bootstrap_mean(
            [s[k].fmeasure for s in s_ar], [s[k].fmeasure for s in s_ncm], n_resamples
        )
        rows.append(_row("xsum", k.replace("rouge", "ROUGE-"), r, scale=100))
    return rows


def _wilson(flags: list[bool]) -> str:
    lo, hi = wilson_interval(sum(flags), len(flags))
    return f"{sum(flags) / len(flags):.3f} [{lo:.3f}, {hi:.3f}]"


def rag_rows() -> list[dict]:
    """Per variant: containment accuracy and exact match (Wilson CI), token F1 (bootstrap CI).

    The NCM variant's row also compares it with the AR variant on the same questions:
    exact McNemar for containment and exact match, paired bootstrap for token F1.
    Files from before exact match / F1 were logged get "-" in those columns.
    """
    rows = []
    for name, (ar_variant, ncm_variant) in RAG_PAIRS.items():
        by_variant: dict[str, dict[str, dict]] = {}
        for x in _jsonl(RESULTS / f"{name}_details.jsonl"):
            by_variant.setdefault(x["variant"], {})[x["id"]] = x
        has_f1 = all("token_f1" in x for v in by_variant.values() for x in v.values())
        for variant, answers in by_variant.items():
            xs = list(answers.values())
            row = {"experiment": name, "variant": variant, "n": len(xs),
                   "contains": _wilson([bool(x["correct"]) for x in xs]),
                   "exact_match": "-", "token_f1": "-",
                   "vs_ar_contains_p": "", "vs_ar_em_p": "", "vs_ar_f1_delta": ""}
            if has_f1:
                row["exact_match"] = _wilson([bool(x["exact_match"]) for x in xs])
                f1 = [x["token_f1"] for x in xs]
                r = paired_bootstrap_mean(f1, f1)
                row["token_f1"] = f"{r.a:.3f} [{r.a_ci[0]:.3f}, {r.a_ci[1]:.3f}]"
            rows.append(row)
        ids = sorted(set(by_variant[ar_variant]) & set(by_variant[ncm_variant]))
        ar = [by_variant[ar_variant][i] for i in ids]
        ncm = [by_variant[ncm_variant][i] for i in ids]
        row = next(r for r in rows if r["experiment"] == name and r["variant"] == ncm_variant)
        t = mcnemar_exact([x["correct"] for x in ar], [x["correct"] for x in ncm])
        row["vs_ar_contains_p"] = f"{t['p_value']:.2g} ({t['only_a']} vs {t['only_b']})"
        if has_f1:
            t = mcnemar_exact([x["exact_match"] for x in ar], [x["exact_match"] for x in ncm])
            row["vs_ar_em_p"] = f"{t['p_value']:.2g} ({t['only_a']} vs {t['only_b']})"
            r = paired_bootstrap_mean([x["token_f1"] for x in ar], [x["token_f1"] for x in ncm])
            row["vs_ar_f1_delta"] = f"{r.delta:+.3f} [{r.delta_ci[0]:+.3f}, {r.delta_ci[1]:+.3f}]"
    return rows


def _md_table(rows: list[dict]) -> list[str]:
    cols = list(rows[0])
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + [
        "| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--resamples", type=int, default=N_RESAMPLES)
    args = ap.parse_args()

    p4, rag = phase4_rows(args.resamples), rag_rows()
    with (RESULTS / "significance.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(p4[0]))
        w.writeheader()
        w.writerows(p4)
    with (RESULTS / "significance_rag.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rag[0]))
        w.writeheader()
        w.writerows(rag)
    md = [
        "# Confidence intervals",
        "",
        "Generated by `python -m ncm_lab.bench.significance`.",
        "",
        f"## Phase 4, full test sets (paired bootstrap, {args.resamples:,} resamples, 95% CI)",
        "",
        *_md_table(p4),
        "",
        "## Phase 5 RAG",
        "",
        "`contains`: a gold answer appears in the reply (lenient). `exact_match`, `token_f1`:",
        "SQuAD scoring. 95% CIs: Wilson for rates, bootstrap for F1. `vs_ar_*` compares the NCM",
        "variant with the AR variant on the same questions: exact McNemar p-values (discordant",
        "pairs AR-only vs NCM-only correct) and the paired-bootstrap F1 difference, NCM − AR.",
        "",
        *_md_table(rag),
        "",
    ]
    (RESULTS / "significance.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
