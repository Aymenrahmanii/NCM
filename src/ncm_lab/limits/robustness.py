"""Limit 5: quality under noisy input (typos, case, missing punctuation).

Real inputs are messy. Both models read the same perturbed text (NCM tokenizes raw text with
its own tokenizer, which Phase 3 showed scores the same as the pre-tokenized data). The score
drop relative to clean input shows how brittle each model is.
"""

import random
import string

from .common import (free_gpu, load_model, load_task, main_metric, model_label, quality, write)


def typos(text: str, rate: float, rng: random.Random) -> str:
    """Swap two adjacent letters in about `rate` of the words longer than 3 letters."""
    out = []
    for w in text.split(" "):
        if len(w) > 3 and w.isalpha() and rng.random() < rate:
            i = rng.randrange(len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
        out.append(w)
    return " ".join(out)


def no_punct(text: str) -> str:
    return " ".join(text.translate(str.maketrans("", "", string.punctuation + "„“”")).split())


def with_typos(rate: float, seed: int = 0):
    def perturb(texts: list[str]) -> list[str]:
        rng = random.Random(seed)  # fresh per call: every model sees the same typos
        return [typos(t, rate, rng) for t in texts]

    return perturb


def perturbations() -> dict:
    return {
        "clean": lambda ts: ts,
        "typos_10pct": with_typos(0.10),
        "typos_30pct": with_typos(0.30),
        "lowercase": lambda ts: [t.lower() for t in ts],
        "uppercase": lambda ts: [t.upper() for t in ts],
        "no_punctuation": lambda ts: [no_punct(t) for t in ts],
    }


def evaluate(model, label: str, task: str, texts: list[str], refs: list[str]) -> list[dict]:
    rows, clean = [], None
    metric = main_metric(task)
    for name, perturb in perturbations().items():
        q = quality(task, model.generate(perturb(texts)), refs)
        clean = q[metric] if name == "clean" else clean
        rows.append({"task": task, "model": label, "perturbation": name, "n": len(texts), **q,
                     "drop_vs_clean": round(clean - q[metric], 2)})
    return rows


def main(n: int = 200, precision: str = "fp16") -> None:
    rows = []
    for task in ("xsum", "wmt14"):
        _, texts, refs = load_task(task, n)
        for kind in ("ncm", "ar"):
            model = load_model(kind, task, precision, batch_size=16)
            rows += evaluate(model, model_label(kind, task), task, texts, refs)
            del model
            free_gpu()
    write("robustness", rows)
