import torch

from ncm_lab.toy_ncm.ar_baseline import ARModel
from ncm_lab.toy_ncm.config import LEVELS, N_CODES, SEQ_LEN, VOCAB_SIZE
from ncm_lab.toy_ncm.generator import Generator, sample
from ncm_lab.toy_ncm.tokenizer import MultiscaleTokenizer


def test_tokenizer_levels_have_expected_lengths():
    tok = MultiscaleTokenizer(layers=1)
    ids = torch.randint(VOCAB_SIZE, (2, SEQ_LEN))
    codes, f_hats, loss = tok.quantize(tok.encode(ids))
    assert [c.shape[1] for c in codes] == list(LEVELS)
    assert len(f_hats) == len(LEVELS)
    assert tok.decode(f_hats[-1]).shape == (2, SEQ_LEN, VOCAB_SIZE)
    assert loss.item() >= 0


def test_codes_rebuild_the_same_canvas():
    tok = MultiscaleTokenizer(layers=1).eval()
    ids = torch.randint(VOCAB_SIZE, (2, SEQ_LEN))
    codes, f_hats, _ = tok.quantize(tok.encode(ids))
    rebuilt = tok.codes_to_fhats(codes)
    assert torch.allclose(rebuilt[-1], f_hats[-1], atol=1e-5)


def test_generator_mask_is_block_causal():
    gen = Generator(layers=1)
    first, last = 0, gen.starts[-1] - 1
    assert gen.mask[last, first]       # finest level sees the coarsest
    assert not gen.mask[first, last]   # coarsest never sees the finest


def test_sample_returns_one_code_per_position():
    tok, gen = MultiscaleTokenizer(layers=1).eval(), Generator(layers=1).eval()
    codes, canvases = sample(gen, tok, batch=3)
    assert [c.shape for c in codes] == [(3, n) for n in LEVELS]
    assert all(int(c.max()) < N_CODES for c in codes)


def test_ar_cache_matches_full_recompute():
    torch.manual_seed(0)
    model = ARModel(layers=1).eval()
    torch.manual_seed(1)
    cached = model.generate(2, temperature=1e-5)
    torch.manual_seed(1)
    full = model.generate(2, temperature=1e-5, use_cache=False)
    assert torch.equal(cached, full)
