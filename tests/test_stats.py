import pytest
import sacrebleu

from ncm_lab.stats import (
    mcnemar_exact,
    paired_bootstrap_bleu,
    paired_bootstrap_mean,
    wilson_interval,
)

REFS = [
    "the cat sat on the mat",
    "a dog runs in the park",
    "it is raining today in the city",
    "she reads a long book every night",
] * 10
GOOD = list(REFS)
BAD = ["the cat", "a dog walks", "rain today", "she reads"] * 10


def test_bleu_point_estimate_matches_sacrebleu():
    r = paired_bootstrap_bleu(BAD, GOOD, REFS, n_resamples=200)
    assert r.a == pytest.approx(sacrebleu.corpus_bleu(BAD, [REFS]).score)
    assert r.b == pytest.approx(100.0)
    assert r.significant and r.delta > 0
    assert r.a_ci[0] <= r.a <= r.a_ci[1]


def test_identical_systems_have_zero_delta():
    r = paired_bootstrap_bleu(BAD, BAD, REFS, n_resamples=200)
    assert r.delta == 0 and r.delta_ci == (0.0, 0.0) and not r.significant


def test_mean_bootstrap():
    a = [0.0, 1.0] * 50
    b = [1.0, 1.0] * 50
    r = paired_bootstrap_mean(a, b, n_resamples=500)
    assert r.a == pytest.approx(0.5) and r.delta == pytest.approx(0.5)
    assert r.significant
    with pytest.raises(ValueError):
        paired_bootstrap_mean([1.0], [1.0, 2.0])


def test_wilson_known_value():
    lo, hi = wilson_interval(35, 40)  # 0.875
    assert lo == pytest.approx(0.7387, abs=1e-3) and hi == pytest.approx(0.9454, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 10  # 10 discordant, all favour b
    r = mcnemar_exact(a, b)
    assert r["only_b"] == 10 and r["p_value"] == pytest.approx(2 / 2**10)
    assert mcnemar_exact(a, a)["p_value"] == 1.0
