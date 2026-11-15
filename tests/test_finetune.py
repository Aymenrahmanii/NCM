import torch
import torch.nn as nn

from ncm_lab.finetune import lora
from ncm_lab.finetune.ncm_loss import target_rows


class FakePre:
    def encode(self, text):
        return [len(w) for w in text.split()]


def test_target_rows_close_with_eos_pad_and_cut():
    rows = target_rows(FakePre(), ["a bb ccc", "a " * 10], length=6, eos=99, pad=0)
    assert rows.tolist()[0] == [1, 2, 3, 99, 0, 0]
    assert rows.tolist()[1] == [1, 1, 1, 1, 1, 99]  # cut to leave room for <eos>


def test_lora_starts_as_base_and_trains_only_its_factors():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 3))
    x = torch.randn(4, 6)
    before = model(x)
    params = lora.add_lora(model, lambda name, lin: name == "0", rank=2, alpha=4)
    assert isinstance(model[0], lora.LoRALinear) and isinstance(model[2], nn.Linear)
    assert torch.allclose(model(x), before)
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in params}


def test_masked_lora_leaves_key_columns_untouched():
    torch.manual_seed(0)
    base = nn.Linear(4, 6)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])  # q | k | v
    layer = lora.LoRALinear(base, rank=2, alpha=2, out_mask=mask)
    nn.init.normal_(layer.B)
    x = torch.randn(3, 4)
    diff = layer(x) - base(x)
    assert torch.all(diff[:, 2:4] == 0) and diff[:, :2].abs().sum() > 0
    assert torch.allclose(layer.weight, base.weight + layer.delta())
