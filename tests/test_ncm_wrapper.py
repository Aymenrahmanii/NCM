"""Wrapper tests: need a GPU and the downloaded checkpoints (scripts/download_ncm.sh)."""

from pathlib import Path

import pytest
import torch

from ncm_lab.models.ncm import ARTIFACTS_DIR, NCMSummarizer, NCMTranslator

pytestmark = pytest.mark.gpu

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _have(task: str) -> bool:
    return (Path(ARTIFACTS_DIR) / task / "generator.pt").exists()


@needs_gpu
@pytest.mark.skipif(not _have("xsum"), reason="xsum checkpoint not downloaded")
def test_summarizer_returns_one_nonempty_string_per_input():
    texts = [
        "The city council voted on Tuesday to build a new bridge over the river. The project "
        "will cost 20 million pounds and is expected to be finished in three years.",
        "Scientists have discovered a new species of frog in the rainforest of Peru.",
    ]
    out = NCMSummarizer().generate(texts)
    assert len(out) == len(texts)
    assert all(isinstance(s, str) and s.strip() for s in out)


@needs_gpu
@pytest.mark.skipif(not _have("wmt14"), reason="wmt14 checkpoint not downloaded")
def test_translator_returns_one_nonempty_string_per_input():
    texts = ["Guten Morgen.", "Das Wetter ist heute schön.", "Ich lerne maschinelles Lernen."]
    out = NCMTranslator().generate(texts)
    assert len(out) == len(texts)
    assert all(isinstance(s, str) and s.strip() for s in out)
