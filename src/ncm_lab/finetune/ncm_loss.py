"""Training loss for the released conditional NCM (XSum checkpoint), rebuilt.

The official repository ships inference code only. The generator uses block-diagonal
attention: each level's positions see the prompt and their own level, and their input is
the canvas of all coarser levels, downsampled to the level's length. So one teacher-forced
pass with the real codes as inputs gives every level's logits at once, exactly the logits
`generate` computes level by level. `greedy_agreement` checks that equivalence on the
released weights before anything is trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def target_rows(pre_tokenizer, texts: list[str], length: int, eos: int, pad: int):
    """GPT-2 ids + <eos>, cut to fit and padded to the tokenizer's canvas length."""
    rows = []
    for t in texts:
        ids = pre_tokenizer.encode(t)[: length - 1] + [eos]
        rows.append(ids + [pad] * (length - len(ids)))
    return torch.tensor(rows)


def teacher_forced_logits(net, prefix: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """(B, prefix_len) prompt ids and (B, L) codes of all levels -> (B, L, vocab) logits."""
    assert net.query_level0 and net.raw_token_prefix and not net.contextual_prefix
    assert not net.use_ctx_encoder and not net.l0_ar and not net.use_post_upsample_conv
    vqvae = net.vqvae_local[0]
    P = net.prefix_len
    with torch.no_grad():  # the VQ tokenizer stays frozen
        z_prefix = vqvae.encoder.wte(prefix[:, :P])
        canvases = vqvae.vq.idx_to_ncm_input(codes)  # inputs of levels 2..n
    q0 = net.build_l0_query_input(prefix.shape[0], z_prefix.dtype)  # level-1 queries (trained)
    x = torch.cat([z_prefix, q0, canvases.to(z_prefix.dtype)], dim=1)
    pad_mask = prefix[:, :P] == net.prefix_pad_id
    return net(x, prefix_pad_mask=pad_mask)


def ncm_loss(net, prefix: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over every code of every level of the target's VQ encoding."""
    with torch.no_grad():
        codes = net.vqvae_local[0].encode(target_ids)
    logits = teacher_forced_logits(net, prefix, codes)
    return F.cross_entropy(logits.flatten(0, 1).float(), codes.flatten())


@torch.no_grad()
def greedy_agreement(net, prefix: torch.Tensor) -> float:
    """Share of greedy-generated codes that the teacher-forced pass reproduces as argmax."""
    codes = net.generate(None, prefix, temperature=0.0, top_k=0, top_p=0.0,
                         use_kv_cache=False)
    pred = teacher_forced_logits(net, prefix, codes).argmax(-1)
    return float((pred == codes).float().mean())
