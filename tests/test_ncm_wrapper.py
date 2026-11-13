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


def _has_cache_fix() -> bool:
    from ncm_lab.models.ncm import NCM_ROOT

    src = NCM_ROOT / "models" / "next_concept.py"
    return src.exists() and "_kv_level" in src.read_text(encoding="utf-8")  # added by PR #5


@needs_gpu
@pytest.mark.skipif(not _have("wmt14"), reason="wmt14 checkpoint not downloaded")
@pytest.mark.skipif(not _has_cache_fix(), reason="NCM_ROOT checkout lacks upstream PR #5")
def test_kv_cache_matches_uncached_in_fp32():
    texts = ["Guten Morgen.", "Das Wetter ist heute schön.", "Ich lerne maschinelles Lernen.",
             "Der Zug kommt um acht Uhr in Berlin an, wenn es keine Verspätung gibt."]
    model = NCMTranslator(precision="fp32", batch_size=4)
    eager = model.generate(texts)
    model.kv_cache = True
    assert model.generate(texts) == eager


@needs_gpu
@pytest.mark.skipif(not _have("wmt14"), reason="wmt14 checkpoint not downloaded")
def test_cast_weights_stores_fp16_and_still_translates():
    model = NCMTranslator(precision="fp16", cast_weights=True)
    dtypes = {p.dtype for p in [*model.net.parameters(), *model.tk.vqvae.parameters()]}
    assert dtypes == {torch.float16}
    out = model.generate(["Guten Morgen.", "Das Wetter ist heute schön."])
    assert len(out) == 2 and all(s.strip() for s in out)


@needs_gpu
@pytest.mark.skipif(not _have("wmt14"), reason="wmt14 checkpoint not downloaded")
def test_compiled_requires_kv_cache():
    with pytest.raises(ValueError, match="kv_cache"):
        NCMTranslator(precision="fp16", compiled=True)
