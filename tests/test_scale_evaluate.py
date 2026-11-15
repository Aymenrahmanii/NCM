import csv

import pytest
import torch

from ncm_lab.scale import evaluate as ev
from ncm_lab.scale import train as strain
from ncm_lab.scale.models import (ARConfig, ARModel, Generator, GeneratorConfig, Tokenizer,
                                  TokenizerConfig, config_dict)

TOK = TokenizerConfig(seq_len=16, levels=(1, 4, 16), n_codes=16, code_dim=8, dim=32, heads=2,
                      enc_layers=1, dec_layers=1)
GEN = GeneratorConfig(dim=32, heads=2, layers=1)
AR = ARConfig(seq_len=16, dim=32, heads=2, layers=1)


class FakeBPE:
    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(strain, "ART", tmp_path)
    monkeypatch.setattr(ev, "ART", tmp_path)
    for name, model, meta in [
        ("tokenizer", Tokenizer(TOK), {"kind": "tokenizer", "config": config_dict(TOK)}),
        ("gen_s0", Generator(GEN, TOK), {"kind": "gen", "config": config_dict(GEN)}),
        ("ar_s0", ARModel(AR), {"kind": "ar", "config": config_dict(AR)}),
    ]:
        (tmp_path / name).mkdir()
        torch.save({"model": model.state_dict(), "meta": meta}, tmp_path / name / "model.pt")
    return tmp_path


def test_finished_runs_skip_the_tokenizer(runs):
    assert set(ev.finished_runs()) == {"gen_s0", "ar_s0"}


@pytest.mark.parametrize("name", ["gen_s0", "ar_s0"])
def test_sampler_is_seeded_and_returns_n_texts(runs, name):
    s = ev.Sampler(name, "cpu")
    a = s.texts(5, 0.8, seed=3, bpe=FakeBPE())
    assert len(a) == 5 and a == s.texts(5, 0.8, seed=3, bpe=FakeBPE())
    assert s.ids(2, 0.0).shape == (2, 16)


def test_plot_writes_png(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    quality = [{"run": "human (test stories)", "kind": "", "mauve": 0.95, "gpt2_large_ppl": 9},
               {"run": "gen25m_s0", "kind": "gen", "mauve": 0.5, "gpt2_large_ppl": 30},
               {"run": "ar7m_s0", "kind": "ar", "mauve": 0.8, "gpt2_large_ppl": 12}]
    lat = [{"run": "gen25m_s0", "median_ms": 20}, {"run": "ar7m_s0", "median_ms": 400}]
    for path, rows in ((tmp_path / "phase10_quality.csv", quality),
                       (tmp_path / "phase10_latency_cuda.csv", lat)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    ev.plot()
    assert (tmp_path / "phase10_pareto.png").stat().st_size > 10_000


class FakeJudge:
    def __init__(self, dev):
        pass

    def features(self, texts):
        return torch.randn(len(texts), 4).numpy()

    def perplexity(self, texts):
        return 10.0

    def entropy(self, texts):
        return 1.0


def test_ceiling_writes_one_row_per_setting(runs, tmp_path, monkeypatch):
    rows = torch.randint(2, 50, (40, 16)).numpy().astype("int16")
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    monkeypatch.setattr(ev, "Judge", FakeJudge)
    monkeypatch.setattr(ev, "load_split", lambda name, **kw: rows)
    monkeypatch.setattr(ev, "load_bpe", FakeBPE)
    monkeypatch.setattr(ev, "decode_ids", lambda bpe, ids: bpe.decode(ids))
    monkeypatch.setattr(ev, "mauve", lambda p, q: 0.5)
    monkeypatch.setattr(ev, "N_TEST", 4)
    monkeypatch.setattr(ev, "DRAWS", 2)
    ev.ceiling("cpu", gen_run="gen_s0")
    with open(tmp_path / "phase10_ceiling.csv", newline="") as f:
        out = list(csv.DictReader(f))
    # 3 levels: reconstruction, last level greedy and sampled, last two sampled, all sampled
    assert [r["run"].split(",")[0] for r in out] == [
        "real levels 1-3", "real levels 1-2", "real levels 1-2", "real levels 1-1",
        "real levels 1-0"]
    assert all(r["n"] == "4" for r in out)


def test_ablation_settings_cover_every_level():
    settings = ev.ablation_settings(9)
    assert all(len(t) == 9 for t, _ in settings.values())
    assert settings["T=0.8, last level greedy"][0][-1] == 0.0
    assert settings["T 1.0 -> 0.5 by level"][0][0] == 1.0
    assert settings["T 1.0 -> 0.5 by level"][0][-1] == 0.5


def test_quality_filter_and_tag(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    monkeypatch.setattr(ev, "Judge", lambda dev: _FakeJudge())
    monkeypatch.setattr(ev, "load_bpe", FakeBPE)
    monkeypatch.setattr(ev, "decode_ids", lambda bpe, ids: bpe.decode(ids))
    monkeypatch.setattr(ev, "reference_texts", lambda split, n, bpe, seed=0: ["w1 w2"] * n)
    monkeypatch.setattr(ev, "mauve", lambda p, q: 0.5)
    monkeypatch.setattr(ev, "N_TUNE", 2)
    monkeypatch.setattr(ev, "N_TEST", 2)
    monkeypatch.setattr(ev, "TEMPS", (1.0,))
    ev.quality("cpu", runs=["ar_s0"], tag="_x")
    with open(tmp_path / "phase10_quality_x.csv", newline="") as f:
        names = [r["run"] for r in csv.DictReader(f)]
    assert names == ["human (test stories)", "ar_s0"]
    assert not (tmp_path / "phase10_quality.csv").exists()


class _FakeJudge:
    def features(self, texts):
        return torch.randn(len(texts), 4).numpy()

    def perplexity(self, texts):
        return 10.0

    def entropy(self, texts):
        return 1.0
