import numpy as np
import pytest
import torch

from ncm_lab.scale import data as sdata
from ncm_lab.scale import train as strain
from ncm_lab.scale.models import (ARConfig, ARModel, Generator, GeneratorConfig, Tokenizer,
                                  TokenizerConfig, ncm_sample)
from ncm_lab.toy_ncm.config import EOS_ID, PAD_ID, VOCAB_SIZE

TINY_TOK = TokenizerConfig(seq_len=16, levels=(1, 2, 4, 8, 16), n_codes=32, code_dim=8,
                           dim=32, heads=2, enc_layers=1, dec_layers=1)
TINY_GEN = GeneratorConfig(dim=32, heads=2, layers=1)


def test_encode_truncates_adds_eos_and_pads():
    out = sdata.encode([[5, 6, 7], list(range(2, 30))], seq_len=8)
    assert out.dtype == np.int16
    assert out[0].tolist() == [5, 6, 7, EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]
    assert out[1].tolist() == list(range(2, 10))  # no room for <eos>


def test_tokenizer_levels_and_rebuilt_canvas():
    tok = Tokenizer(TINY_TOK).eval()
    ids = torch.randint(VOCAB_SIZE, (2, 16))
    codes, f_hats, _ = tok.quantize(tok.encode(ids))
    assert [c.shape[1] for c in codes] == list(TINY_TOK.levels)
    assert torch.allclose(tok.codes_to_fhats(codes)[-1], f_hats[-1], atol=1e-5)
    assert tok.decode(f_hats[-1]).shape == (2, 16, VOCAB_SIZE)


def test_generator_is_block_causal_and_sample_keeps_prefix():
    tok, gen = Tokenizer(TINY_TOK).eval(), Generator(TINY_GEN, TINY_TOK).eval()
    assert gen.mask[-1, 0] and not gen.mask[0, -1]
    real, _, _ = tok.quantize(tok.encode(torch.randint(VOCAB_SIZE, (3, 16))))
    codes, canvases = ncm_sample(gen, tok, 3, temperature=0.7, prefix=real[:2])
    assert [c.shape for c in codes] == [(3, n) for n in TINY_TOK.levels]
    assert all(torch.equal(c, r) for c, r in zip(codes[:2], real[:2]))
    _, canvases = ncm_sample(gen, tok, 3, prefix=real)
    assert torch.allclose(canvases[-1], tok.codes_to_fhats(real)[-1])


def test_ar_greedy_cache_matches_full_forward():
    torch.manual_seed(0)
    model = ARModel(ARConfig(seq_len=12, dim=32, heads=2, layers=2)).eval()
    out = model.generate(2, temperature=0.0)
    seq = torch.full((2, 1), EOS_ID)
    for _ in range(12):
        seq = torch.cat([seq, model(seq)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(out, seq[:, 1:])


def test_batch_indices_depend_only_on_seed_and_step():
    a = strain.batch_indices(1000, seed=1, step=7)
    assert np.array_equal(a, strain.batch_indices(1000, seed=1, step=7))
    assert not np.array_equal(a, strain.batch_indices(1000, seed=1, step=8))


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    rows = np.random.default_rng(0).integers(2, VOCAB_SIZE, (200, 16)).astype(np.int16)
    monkeypatch.setattr(strain, "load_split", lambda name, **kw: rows)
    monkeypatch.setattr(strain, "ART", tmp_path)
    monkeypatch.setattr(strain, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(strain, "BATCH", 8)
    monkeypatch.setattr(strain, "TOKENS_PER_STEP", 8 * 16)
    val = torch.from_numpy(rows[:16].astype(np.int64))

    def go(name: str, stop_after: int | None = None) -> Tokenizer:
        torch.manual_seed(0)
        tok = Tokenizer(TINY_TOK)

        def on_step(step: int, total: int) -> None:
            if stop_after is not None and step + 1 == stop_after:
                (tmp_path / "STOP").touch()

        done = strain.run(name, tok, lambda ids, s: strain.tokenizer_loss(tok, ids),
                          lambda: strain.tokenizer_eval(tok, val), {"kind": "tokenizer"},
                          tokens=6 * 8 * 16, seed=3, save_every=2, eval_every=3,
                          on_step=on_step, dev="cpu")
        return tok, done
    return go, tmp_path


def test_stopped_and_resumed_run_matches_uninterrupted(tiny_run):
    go, root = tiny_run
    straight, done = go("a")
    assert done and (root / "a" / "model.pt").exists()
    _, done = go("b", stop_after=4)
    assert not done and (root / "b" / "ckpt.pt").exists()
    (root / "STOP").unlink()
    resumed, done = go("b")
    assert done and not (root / "b" / "ckpt.pt").exists()
    for p, q in zip(straight.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(p, q)
