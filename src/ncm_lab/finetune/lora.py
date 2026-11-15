"""Minimal LoRA (Hu et al. 2021), written here instead of adding `peft` as a dependency.

The NCM fuses query, key and value into one `c_attn` projection; "LoRA on query and value"
there means a LoRA update on `c_attn` whose key columns are masked to zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """base(x) + scale * B(A(x)), with base frozen and B zero-initialised (starts as base)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 out_mask: torch.Tensor | None = None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.register_buffer("out_mask", out_mask, persistent=False)

    @property
    def weight(self) -> torch.Tensor:  # some callers read .weight directly
        return self.base.weight + self.delta()

    @property
    def bias(self):
        return self.base.bias

    def delta(self) -> torch.Tensor:
        d = self.scale * (self.B @ self.A)
        return d if self.out_mask is None else d * self.out_mask[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)
        if self.out_mask is not None:
            h = h * self.out_mask.to(h.dtype)
        return self.base(x) + self.scale * h


Selector = Callable[[str, nn.Linear], "torch.Tensor | bool"]


def add_lora(model: nn.Module, select: Selector, rank: int, alpha: float) -> list[nn.Parameter]:
    """Freeze `model`, wrap every Linear that `select` accepts, return the trainable params.

    `select(name, linear)` returns False to skip, True for a plain LoRA, or an output mask.
    """
    model.requires_grad_(False)
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    params: list[nn.Parameter] = []
    for name, lin in targets:
        choice = select(name, lin)
        if choice is False:
            continue
        mask = None if choice is True else choice.to(lin.weight.device, lin.weight.dtype)
        wrapped = LoRALinear(lin, rank, alpha, mask).to(lin.weight.device)
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, attr, wrapped)
        params += [wrapped.A, wrapped.B]
    if not params:
        raise ValueError("add_lora matched no Linear layer")
    return params


def ncm_qv_selector(n_embd: int, kv_dim: int) -> Selector:
    """LoRA on the query and value parts of the generator's fused c_attn."""
    mask = torch.cat([torch.ones(n_embd), torch.zeros(kv_dim), torch.ones(kv_dim)])

    def select(name: str, lin: nn.Linear):
        return mask if name.endswith("attn.c_attn") and name.startswith("transformer.blocks") \
            else False
    return select


def ncm_all_selector(name: str, lin: nn.Linear) -> bool:
    """Every attention and MLP projection of the generator's transformer blocks."""
    return name.startswith("transformer.blocks") and name.rsplit(".", 1)[-1] in {
        "c_attn", "c_proj", "c_fc", "w_gate", "w_up"}


def bart_qv_selector(name: str, lin: nn.Linear) -> bool:
    return name.endswith(("q_proj", "v_proj"))


def bart_all_selector(name: str, lin: nn.Linear) -> bool:
    return name.endswith(("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"))
