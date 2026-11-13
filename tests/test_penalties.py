"""The incremental penalty loop must pick exactly the tokens upstream's loop picks."""

import pytest
import torch

from ncm_lab.models.ncm import pick_tokens


def reference_pick(logits, freq, mu, adj, allowance):
    """Copy of pick_tokens in third_party/ncm/scripts/eval_shard.py."""
    if freq <= 0 and not adj:
        return logits.argmax(-1)
    b, t_len, v = logits.shape
    lg = logits.double()
    allow = (allowance[:v] * t_len)[None, :] if allowance is not None else None
    al = adj[0] if adj else 0.0
    aw = int(adj[1]) if adj and len(adj) > 1 else 1
    ad = adj[2] if adj and len(adj) > 2 else 1.0
    cnt = torch.zeros(b, v, dtype=torch.float64)
    out = torch.empty(b, t_len, dtype=torch.long)
    hist = []
    for t in range(t_len):
        step = lg[:, t, :].clone()
        if freq > 0:
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= freq * excess
            if mu > 0 and allow is not None:
                step += mu * ((t_len - t) / t_len) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((b, 1), -al * ad**j, dtype=torch.float64))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(b, 1, dtype=torch.float64))
    return out


def _peaked_logits(b, t_len, v, seed):
    # few strong candidates per row, so penalties actually change the argmax
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(b, t_len, v, generator=g)
    lg[..., :8] += 6.0
    return lg.half().float()  # values on the fp16 grid, like the decoder's output


XSUM = dict(freq=2.0, mu=1.0, adj=[5.0, 4.0, 0.5])
CASES = {
    "xsum": XSUM,
    "freq_only": dict(freq=2.0, mu=0.0, adj=None),
    "adj_only": dict(freq=0.0, mu=0.0, adj=[5.0, 2.0, 1.0]),
    "no_allowance": dict(freq=1.5, mu=1.0, adj=[3.0], allowance=None),
    "none": dict(freq=0.0, mu=0.0, adj=None),
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference(name, seed):
    v = 300
    g = torch.Generator().manual_seed(100 + seed)
    unigram = torch.rand(v + 5, generator=g, dtype=torch.float64)
    kw = {"allowance": unigram / unigram.sum(), **CASES[name]}
    logits = _peaked_logits(4, 48, v, seed)
    assert torch.equal(pick_tokens(logits, **kw), reference_pick(logits, **kw))


def test_penalties_change_the_picks():
    logits = _peaked_logits(4, 48, 300, 0)
    unigram = torch.full((300,), 1 / 300, dtype=torch.float64)
    assert not torch.equal(pick_tokens(logits, allowance=unigram, **XSUM), logits.argmax(-1))
