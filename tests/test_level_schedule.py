"""The random-weight OWT architecture is timed at shorter canvases by cutting its schedule."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ncm_lab.models.ncm import _truncate_levels

OWT = Path("third_party/ncm-pr5-pr4/config/eval/owt.yaml")
pytestmark = pytest.mark.skipif(not OWT.exists(), reason="upstream checkout not present")


def _levels(cfg):
    return list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)


@pytest.mark.parametrize("length,expected_last_two", [(256, [225, 256]), (512, [484, 512]),
                                                      (1024, [961, 1024])])
def test_cut_keeps_lower_levels_and_ends_at_length(length, expected_last_two):
    cfg = OmegaConf.load(OWT)
    full = _levels(cfg)
    _truncate_levels(cfg, length)
    levels = _levels(cfg)
    assert levels[-2:] == expected_last_two
    assert levels[:-1] == [v for v in full if v < length]
    n = len(levels)
    assert len(cfg.tokenizer.vqvae.vqvae_vocab_size) == n
    assert len(cfg.model.vocab_size) == n - 1  # level 0 is sampled, not predicted
    assert cfg.tokenizer.vqvae.context_length == length
    assert cfg.model.context_length == length + 1


def test_full_length_is_unchanged():
    cfg = OmegaConf.load(OWT)
    full = _levels(cfg)
    _truncate_levels(cfg, 1024)
    assert _levels(cfg) == full and len(full) == 32
