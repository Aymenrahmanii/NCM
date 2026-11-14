"""Phase 10 diagnostic: which level of the toy NCM breaks coherence?

    python -m ncm_lab.toy_ncm.diagnose

The tokenizer reconstructs 98.7% of tokens, so the incoherence of samples (GPT-2 perplexity
377 vs 13 for real stories) comes from the generator. For each k = 0..6, keep a real test
story's codes for levels 1..k and let the generator write levels k+1..6, then decode:

- k = 6 is the tokenizer's reconstruction (upper bound), k = 0 is a free sample.
- The k where perplexity jumps is the level whose generation breaks the text.
- Each completion runs twice, sampled (temperature 1, as in Phase 2) and greedy. If greedy
  closes most of the gap, the damage comes from sampling every position of a level
  independently, not from what the model knows.
- A last-level temperature sweep (after real and after generated coarser levels) tests the
  cheapest decoding fix.

Also reports teacher-forced accuracy per level: with the real coarser levels as input, how
often the generator's top code equals the real one, and its loss (nats per code).
Writes results/phase10_diag_levels.csv, _teacher_forced.csv, _samples.txt and _summary.md.
"""

from __future__ import annotations

import csv

import torch
import torch.nn.functional as F

from .config import LEVELS, PAD_ID, RESULTS_DIR, device
from .data import decode_ids, load_bpe, load_tokens
from .evaluate import distinct_2, gpt2_perplexity, load_judge
from .generator import Generator, load_generator, sample
from .tokenizer import MultiscaleTokenizer, load_tokenizer

N_STORIES = 512
BATCH = 128
MODES = {"sampled": 1.0, "greedy": 0.0}
# Per-level temperature (a planned Phase 10 decoding ablation): cool only the last level,
# after real or after generated coarser levels.
LAST_LEVEL_TEMPS = (0.7, 0.5, 0.3)


def grid() -> list[tuple[str, int, float | list[float]]]:
    """(mode label, real levels kept, temperature) for every row of the table."""
    n = len(LEVELS)
    rows = [(mode, k, t) for mode, t in MODES.items() for k in range(n + 1)
            if not (k == n and mode == "greedy")]  # nothing left to generate at k = n
    for k in (n - 1, 0):
        rows += [(f"last level T={t}", k, [1.0] * (n - 1) + [t]) for t in LAST_LEVEL_TEMPS]
    rows.append(("all levels T=0.7", 0, 0.7))
    return rows


@torch.no_grad()
def complete(gen: Generator, tok: MultiscaleTokenizer, ids: torch.Tensor, k: int,
             temperature: float | list[float]) -> torch.Tensor:
    """Token ids after keeping the real codes of levels 1..k and generating the rest."""
    codes, _, _ = tok.quantize(tok.encode(ids))
    _, canvases = sample(gen, tok, len(ids), temperature, prefix=codes[:k])
    return tok.decode(canvases[-1]).argmax(-1)


def token_agreement(out: torch.Tensor, ids: torch.Tensor) -> float:
    """Share of the real story's non-pad tokens reproduced at the same position."""
    keep = ids != PAD_ID
    return ((out == ids) & keep).sum().item() / keep.sum().item()


@torch.no_grad()
def teacher_forced(gen: Generator, tok: MultiscaleTokenizer, data: torch.Tensor,
                   dev: str) -> list[dict]:
    hits = torch.zeros(len(LEVELS))
    nll = torch.zeros(len(LEVELS))
    for i in range(0, len(data), BATCH):
        ids = data[i : i + BATCH].to(dev)
        codes, f_hats, _ = tok.quantize(tok.encode(ids))
        logits = gen([torch.zeros_like(f_hats[0])] + f_hats[:-1])
        for k, (lg, c) in enumerate(zip(logits, codes)):
            hits[k] += (lg.argmax(-1) == c).float().mean().item() * len(ids)
            nll[k] += F.cross_entropy(lg.flatten(0, 1).float(), c.flatten()).item() * len(ids)
    n = len(data)
    return [{"level": k + 1, "length": LEVELS[k], "code_accuracy": round(hits[k].item() / n, 4),
             "nll_nats": round(nll[k].item() / n, 3)} for k in range(len(LEVELS))]


def main() -> None:
    torch.manual_seed(0)
    dev = device()
    bpe = load_bpe()
    _, test = load_tokens()
    data = test[:N_STORIES]
    tok, gen, judge = load_tokenizer(dev), load_generator(dev), load_judge(dev)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    RESULTS_DIR.mkdir(exist_ok=True)

    real_texts = [decode_ids(bpe, r.tolist()) for r in data]
    rows, examples = [], {}
    with torch.autocast(**amp):
        tf = teacher_forced(gen, tok, data, dev)
        for mode, k, temp in grid():
            torch.manual_seed(1000 + k)
            outs = torch.cat([complete(gen, tok, data[i : i + BATCH].to(dev), k, temp)
                              for i in range(0, len(data), BATCH)])
            texts = [decode_ids(bpe, r.tolist()) for r in outs]
            rows.append({"mode": mode, "real_levels": k, "gen_levels": len(LEVELS) - k,
                         "gpt2_ppl": round(gpt2_perplexity(texts, dev, judge), 1),
                         "token_agreement": round(token_agreement(outs.cpu(), data), 4),
                         "distinct_2": round(distinct_2(texts), 3)})
            examples[(mode, k)] = texts[:3]
            print(rows[-1], flush=True)
        real_ppl = gpt2_perplexity(real_texts, dev, judge)

    _write_csv(RESULTS_DIR / "phase10_diag_levels.csv", rows)
    _write_csv(RESULTS_DIR / "phase10_diag_teacher_forced.csv", tf)
    with open(RESULTS_DIR / "phase10_diag_samples.txt", "w", encoding="utf-8") as f:
        for j in range(3):
            f.write(f"===== story {j + 1} =====\nreal: {real_texts[j]}\n")
            for (mode, k), texts in examples.items():
                f.write(f"{mode}, real levels 1..{k}: {texts[j]}\n")
            f.write("\n")
    _write_summary(rows, tf, real_ppl)


def _write_csv(path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_summary(rows: list[dict], tf: list[dict], real_ppl: float) -> None:
    lines = ["# Phase 10 diagnostic: where the toy NCM loses coherence", "",
             f"{N_STORIES} test stories, 64 tokens. Real stories: GPT-2 perplexity "
             f"{real_ppl:.1f}.", "",
             "| real levels kept | generated | mode | GPT-2 ppl | token agreement | distinct-2 |",
             "|---|---|---|---|---|---|"]
    lines += [f"| 1..{r['real_levels']} | {r['gen_levels']} | {r['mode']} | {r['gpt2_ppl']} | "
              f"{r['token_agreement']:.1%} | {r['distinct_2']} |" for r in rows]
    lines += ["", "Teacher-forced (real coarser levels as input):", "",
              "| level | length | top-1 code accuracy | loss (nats/code) |", "|---|---|---|---|"]
    lines += [f"| {r['level']} | {r['length']} | {r['code_accuracy']:.1%} | {r['nll_nats']} |"
              for r in tf]
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "phase10_diag_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
