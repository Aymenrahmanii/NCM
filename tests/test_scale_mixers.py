import pytest
import torch
import torch.nn.functional as F

from ncm_lab.scale.models import (MIXERS, Generator, GeneratorConfig, Tokenizer,
                                  TokenizerConfig, ncm_sample)

TOK = TokenizerConfig(seq_len=16, levels=(1, 2, 4, 8, 16), n_codes=32, code_dim=8,
                      dim=32, heads=2, enc_layers=1, dec_layers=1)


def make(mixer: str) -> Generator:
    torch.manual_seed(0)
    return Generator(GeneratorConfig(dim=32, heads=2, layers=2, mixer=mixer, conv_from=8,
                                     conv_kernel=3), TOK).eval()


def canvases(n: int) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.randn(2, 16, 8) for _ in range(n)]


@pytest.mark.parametrize("mixer", MIXERS)
def test_levels_never_see_finer_levels(mixer):
    gen, cs = make(mixer), canvases(5)
    full = gen(cs)
    for k in range(1, 5):
        part = gen(cs[:k])
        for i in range(k):
            assert torch.allclose(part[i], full[i], atol=1e-5), (mixer, k, i)


@pytest.mark.parametrize("mixer", ["nocross", "conv"])
def test_isolated_levels_ignore_coarser_inputs(mixer):
    gen, cs = make(mixer), canvases(5)
    base = gen(cs)
    changed = gen([cs[0] + 1.0] + cs[1:])  # only level 1's input changes
    isolated = range(1, 5) if mixer == "nocross" else range(3, 5)  # conv: levels of >= 8
    for i in isolated:
        assert torch.allclose(base[i], changed[i], atol=1e-5), (mixer, i)
    assert not torch.allclose(base[0], changed[0])


def test_linear_attention_matches_a_naive_loop():
    gen = make("linear")
    mix = gen.transformer.blocks[0].mix
    x = torch.randn(2, 15, 32)
    out = mix(x, 4)
    b, t, c = x.shape
    q, k, v = mix.qkv(x).view(b, t, 3, 2, 16).permute(2, 0, 3, 1, 4)
    q, k = F.elu(q) + 1, F.elu(k) + 1
    ids = mix.level_ids[:t]
    ref = torch.empty(b, 2, t, 16)
    for i in range(t):
        seen = ids <= ids[i]
        w = (q[:, :, i:i + 1] * k[:, :, seen]).sum(-1)  # (b, h, s)
        ref[:, :, i] = (w[..., None] * v[:, :, seen]).sum(2) / w.sum(-1, keepdim=True)
    ref = mix.proj(ref.transpose(1, 2).reshape(b, t, c))
    assert torch.allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize("mixer", MIXERS)
def test_every_mixer_samples(mixer):
    tok = Tokenizer(TOK).eval()
    codes, _ = ncm_sample(make(mixer), tok, 3, temperature=0.8)
    assert [c.shape for c in codes] == [(3, n) for n in TOK.levels]


def test_run_names_keep_phase10_and_separate_variants():
    from ncm_lab.scale.train import DEFAULT_TOKENS, generator_run_name

    assert generator_run_name("gen25m", "attn", DEFAULT_TOKENS["gen"], 0) == "gen25m_s0"
    assert generator_run_name("gen25m", "attn", 100e6, 1) == "gen25m-attn-100m_s1"
    assert generator_run_name("gen25m", "conv", 200e6, 0) == "gen25m-conv-200m_s0"


def test_speed_bench_runs_on_cpu():
    from ncm_lab.scale import mixer_speed

    assert mixer_speed.tokenizer_config(1024).levels[-1] == 1024
    rows = mixer_speed.bench("cpu", repeats=1, lengths=(16,), size="gen12m", warmup=0)
    assert [r["mixer"] for r in rows] == list(MIXERS) and all(r["median_ms"] > 0 for r in rows)
