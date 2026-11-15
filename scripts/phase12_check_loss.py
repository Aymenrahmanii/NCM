"""Check the rebuilt NCM loss on the released XSum checkpoint before training with it.

    NCM_ROOT=third_party/ncm-pr5-pr4 PYTHONPATH=src python scripts/phase12_check_loss.py

1. Greedy agreement: the teacher-forced pass must reproduce greedy generation's codes.
2. Loss on the model's own greedy codes (should be lowest) vs on real SAMSum summaries
   (the zero-shot starting point) vs on summaries paired with the wrong dialogue.
"""

from __future__ import annotations

import csv
import json

import torch
import torch.nn.functional as F

from ncm_lab.finetune.ncm_loss import (greedy_agreement, ncm_loss, target_rows,
                                       teacher_forced_logits)
from ncm_lab.models.ncm import NCMSummarizer

N = 64


def main() -> None:
    m = NCMSummarizer(device="cuda", precision="fp32", kv_cache=False)
    with open("data/samsum/raw/validation.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:N]
    prefix = m.prepare([r["dialogue"] for r in rows])
    targets = target_rows(m.tk.pre_tokenizer, [r["summary"] for r in rows],
                          m.tk.vqvae.context_length, m.eos, m.pad).to(m.device)
    out = {}
    agree = [greedy_agreement(m.net, p) for p in prefix.split(16)]
    out["greedy_agreement"] = round(sum(agree) / len(agree), 5)
    with torch.no_grad():
        greedy_codes = m.net.generate(None, prefix[:16], temperature=0.0, top_k=0, top_p=0.0)
        logits = teacher_forced_logits(m.net, prefix[:16], greedy_codes)
        out["loss_own_greedy_codes"] = round(
            F.cross_entropy(logits.flatten(0, 1), greedy_codes.flatten()).item(), 4)
        out["loss_samsum"] = round(ncm_loss(m.net, prefix, targets).item(), 4)
        out["loss_samsum_shuffled"] = round(
            ncm_loss(m.net, prefix, targets.roll(1, dims=0)).item(), 4)
    print(json.dumps(out, indent=1))
    with open("results/phase12_loss_check.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
