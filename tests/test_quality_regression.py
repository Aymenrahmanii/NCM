"""Fails if NCM summarization quality drops below a threshold on 20 fixed XSum examples.

Threshold = score measured on 2026-10-26 (RTX 3050, fp16) minus a small margin, so changes
to the wrapper (tokenization, decoding, precision) that hurt quality are caught.
"""

import pickle
from pathlib import Path

import pytest
import torch

from ncm_lab.models.ncm import ARTIFACTS_DIR

ROUGE_L_THRESHOLD = 22.0
N = 20

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (Path(ARTIFACTS_DIR) / "xsum" / "generator.pt").exists(),
                       reason="xsum checkpoint not downloaded"),
]


def test_xsum_rouge_l_does_not_regress():
    from rouge_score import rouge_scorer

    from ncm_lab.models.ncm import NCMSummarizer

    passages = pickle.load(open("data/xsum/passages.pickle", "rb"))[:N]
    refs = pickle.load(open("data/xsum/reference_summaries.pickle", "rb"))[:N]
    hyps = NCMSummarizer(precision="fp16").generate(passages)
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = 100 * sum(sc.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)) / N
    print(f"ROUGE-L on {N} fixed examples: {rouge_l:.2f}")
    assert rouge_l >= ROUGE_L_THRESHOLD
