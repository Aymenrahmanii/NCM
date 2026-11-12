"""Statistics harness for the research plan's measurement protocol.

- Paired bootstrap for corpus BLEU (resamples sentence-level sufficient statistics, so
  10,000 resamples of 3,003 sentences take seconds) and for any per-example metric whose
  corpus score is a mean (ROUGE, exact match, token F1).
- Wilson score interval for proportions and an exact McNemar test for paired binary outcomes.

A difference counts only if its 95% interval excludes zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 10_000


@dataclass(frozen=True)
class PairedResult:
    a: float
    b: float
    a_ci: tuple[float, float]
    b_ci: tuple[float, float]
    delta: float  # b - a
    delta_ci: tuple[float, float]
    n: int

    @property
    def significant(self) -> bool:
        lo, hi = self.delta_ci
        return lo > 0 or hi < 0


def _ci(samples: np.ndarray, level: float) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return float(lo), float(hi)


def _resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def paired_bootstrap_mean(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> PairedResult:
    """Paired bootstrap for a metric whose corpus value is the mean of per-example scores."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("scores_a and scores_b must be equal-length, non-empty 1-D sequences")
    idx = _resample_indices(len(a), n_resamples, seed)
    ma, mb = a[idx].mean(axis=1), b[idx].mean(axis=1)
    return PairedResult(
        a=float(a.mean()), b=float(b.mean()),
        a_ci=_ci(ma, level), b_ci=_ci(mb, level),
        delta=float(b.mean() - a.mean()), delta_ci=_ci(mb - ma, level),
        n=len(a),
    )


def _bleu_stats(hyps: Sequence[str], refs: Sequence[str]) -> tuple[np.ndarray, object]:
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    # Rows: [sys_len, ref_len, correct_1..4, total_1..4], one per sentence.
    stats = np.asarray(bleu._extract_corpus_statistics(list(hyps), [list(refs)]), dtype=np.int64)
    return stats, bleu


def _bleu_from_sums(sums: np.ndarray, bleu) -> float:
    s = sums.tolist()
    order = bleu.max_ngram_order
    return bleu.compute_bleu(
        correct=s[2 : 2 + order], total=s[2 + order : 2 + 2 * order],
        sys_len=s[0], ref_len=s[1],
        smooth_method=bleu.smooth_method, smooth_value=bleu.smooth_value,
        effective_order=bleu.effective_order, max_ngram_order=order,
    ).score


def paired_bootstrap_bleu(
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    refs: Sequence[str],
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
    chunk: int = 500,
) -> PairedResult:
    """Paired bootstrap for sacrebleu corpus BLEU (default settings), same resamples for both."""
    if not (len(hyps_a) == len(hyps_b) == len(refs)) or not refs:
        raise ValueError("hyps_a, hyps_b and refs must be equal-length and non-empty")
    sa, bleu = _bleu_stats(hyps_a, refs)
    sb, _ = _bleu_stats(hyps_b, refs)
    idx = _resample_indices(len(refs), n_resamples, seed)
    ba, bb = np.empty(n_resamples), np.empty(n_resamples)
    for start in range(0, n_resamples, chunk):  # chunked to bound memory on long corpora
        sl = idx[start : start + chunk]
        sums_a, sums_b = sa[sl].sum(axis=1), sb[sl].sum(axis=1)
        for j in range(len(sl)):
            ba[start + j] = _bleu_from_sums(sums_a[j], bleu)
            bb[start + j] = _bleu_from_sums(sums_b[j], bleu)
    a, b = _bleu_from_sums(sa.sum(axis=0), bleu), _bleu_from_sums(sb.sum(axis=0), bleu)
    return PairedResult(
        a=a, b=b, a_ci=_ci(ba, level), b_ci=_ci(bb, level),
        delta=b - a, delta_ci=_ci(bb - ba, level), n=len(refs),
    )


def bootstrap_quantile(
    samples: Sequence[float],
    q: float = 0.5,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """A quantile of the samples (e.g. median latency) and its bootstrap interval."""
    x = np.asarray(samples, float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("samples must be a non-empty 1-D sequence")
    boot = np.quantile(x[_resample_indices(len(x), n_resamples, seed)], q, axis=1)
    return float(np.quantile(x, q)), _ci(boot, level)


def paired_ratio_of_medians(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """median(a) / median(b) with a paired bootstrap interval.

    Pairs are calls made in the same interleaved round, so drift (clock, temperature) that
    hits both models in a round cancels out of the interval.
    """
    a, b = np.asarray(samples_a, float), np.asarray(samples_b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("samples must be equal-length, non-empty 1-D sequences")
    idx = _resample_indices(len(a), n_resamples, seed)
    ratios = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    return float(np.median(a) / np.median(b)), _ci(ratios, level)


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need n > 0 and 0 <= successes <= n")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """Exact two-sided McNemar test (binomial on the discordant pairs)."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired outcomes must have equal length")
    only_a = sum(bool(x) and not bool(y) for x, y in zip(correct_a, correct_b))
    only_b = sum(bool(y) and not bool(x) for x, y in zip(correct_a, correct_b))
    m = only_a + only_b
    if m == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2**m
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}
