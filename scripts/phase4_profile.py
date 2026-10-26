"""Where does NCM spend its time? Split one generation into its stages.

    python scripts/phase4_profile.py
"""

import pickle
import time
from pathlib import Path

import torch

from ncm_lab.models.ncm import NCMSummarizer, NCMTranslator


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


@torch.no_grad()
def profile(model, items, bs: int, repeats: int = 3) -> dict:
    x = model.prepare(items[:bs])
    stages = {"level passes (net.generate)": 0.0, "codes -> vectors -> logits": 0.0,
              "pick tokens (penalties)": 0.0}
    for r in range(repeats + 1):  # first round = warm-up
        with model._autocast():
            codes, a = timed(lambda: model.net.generate(
                None, x, temperature=model.temperature, top_k=model.top_k, top_p=model.top_p,
                cfg_scale=model.cfg_scale, cfg_null_eos=model.eos))
            logits, b = timed(lambda codes=codes: model.tk.vqvae.decoder(
                model.tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)[1][-1]).float())
            _, c = timed(lambda logits=logits: model._pick_tokens(logits))
        if r:
            for k, v in zip(stages, (a, b, c)):
                stages[k] += v / repeats
    return stages


def main() -> None:
    lines = ["# Phase 4: where NCM spends its time (fp16, ms per batch)", ""]
    xsum = pickle.load(open("data/xsum/passages.pickle", "rb"))[:32]
    wmt = pickle.load(open("data/wmt14/sources.pickle", "rb"))[:32]
    for cls, items in ((NCMSummarizer, xsum), (NCMTranslator, wmt)):
        model = cls(precision="fp16")
        lines += [f"## {model.task}", "", "| stage | batch 1 | batch 32 |", "|---|---|---|"]
        p1, p32 = profile(model, items, 1), profile(model, items, 32)
        for k in p1:
            lines.append(f"| {k} | {p1[k]:.1f} | {p32[k]:.1f} |")
        lines.append(f"| **total** | {sum(p1.values()):.1f} | {sum(p32.values()):.1f} |")
        lines.append("")
        del model
        torch.cuda.empty_cache()
    Path("results/phase4_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
