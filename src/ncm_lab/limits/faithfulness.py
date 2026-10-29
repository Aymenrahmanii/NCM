"""Limit 4: how often outputs invent numbers or names, or repeat themselves (no GPU needed).

Scores the saved Phase 4 outputs (results/phase4_outputs_<task>_<model>.jsonl) against their
sources. Simple, transparent proxies rather than a learned metric:

- invented number: a number in the output that is not in the source (for translation, not in
  the source or the reference either, since numbers can be written differently);
- invented name: a capitalized word that is not the first word of a sentence and does not
  appear in the source (or, for translation, the reference);
- repetition: the same word twice in a row, or a repeated word bigram ("coup d coup coup").

The proxies over-count a little (a summary may legitimately say "Tuesday" when the article
said "yesterday"), equally for every model, so compare models with each other rather than
reading the numbers as absolute hallucination rates.
"""

import json
import re

from .common import RESULTS, write

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
WORD = re.compile(r"[A-Za-zÀ-ÿ][\w'’-]*")
ALWAYS_CAPITALIZED = {"I", "Mr", "Mrs", "Ms", "Dr", "Sir", "St"}


def numbers(text: str) -> set[str]:
    return {re.sub(r"[.,]", "", m) for m in NUMBER.findall(text)}


def names(text: str) -> set[str]:
    """Capitalized words that are not the first word of a sentence."""
    out = set()
    for sentence in re.split(r"(?<=[.!?:;])\s+|\"|“|”", text):
        for i, w in enumerate(WORD.findall(sentence)):
            if i > 0 and w[0].isupper() and w not in ALWAYS_CAPITALIZED:
                out.add(w.strip("'’-"))
    return out


def repeats(text: str) -> bool:
    toks = [t.lower() for t in WORD.findall(text)]
    if any(a == b for a, b in zip(toks, toks[1:])):
        return True
    bigrams = list(zip(toks, toks[1:]))
    return len(bigrams) != len(set(bigrams))


def check(hyp: str, support: str) -> dict:
    support_lower = support.lower()
    invented_numbers = sorted(numbers(hyp) - numbers(support))
    invented_names = sorted(n for n in names(hyp) if n.lower() not in support_lower)
    return {"invented_numbers": invented_numbers, "invented_names": invented_names,
            "repeats": repeats(hyp)}


def score(model: str, task: str, hyps: list[str], supports: list[str]) -> tuple[dict, list]:
    checks = [check(h, s) for h, s in zip(hyps, supports)]
    n = len(checks)
    row = {"task": task, "model": model, "n": n,
           "share_invented_number": round(sum(bool(c["invented_numbers"]) for c in checks) / n, 3),
           "share_invented_name": round(sum(bool(c["invented_names"]) for c in checks) / n, 3),
           "share_repetition": round(sum(c["repeats"] for c in checks) / n, 3),
           "share_any": round(sum(bool(c["invented_numbers"] or c["invented_names"]
                                       or c["repeats"]) for c in checks) / n, 3)}
    flagged = [{"task": task, "model": model, "hyp": h} | c for h, c in zip(hyps, checks)
               if c["invented_numbers"] or c["invented_names"] or c["repeats"]]
    return row, flagged[:25]


def sources(task: str, n: int) -> list[str]:
    from .common import load_task

    _, texts, refs = load_task(task, n)
    if task == "wmt14":  # a translation may use the reference's spelling of a name/number
        return [s + "\n" + r for s, r in zip(texts, refs)]
    return texts  # xsum: only the article (the 512-token prefix both models read)


def main(n: int = 500, precision: str = "fp16") -> None:
    rows, details = [], []
    for task in ("xsum", "wmt14"):
        support = None
        for model in ("ncm", "ar"):
            path = RESULTS / f"phase4_outputs_{task}_{model}.jsonl"
            if not path.exists():
                print(f"skip {path}: run phase 4 first")
                continue
            hyps = [json.loads(line)["hyp"] for line in open(path, encoding="utf-8")]
            support = support or sources(task, len(hyps))
            row, flagged = score("NCM" if model == "ncm" else "AR baseline", task, hyps,
                                 support[: len(hyps)])
            rows.append(row)
            details += flagged
    write("faithfulness", rows, details)
