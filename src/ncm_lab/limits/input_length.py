"""Limit 1: translation quality as the input gets longer (WMT14 De->En).

NCM reads a fixed-size prefix (96 tokens for the WMT14 checkpoint) and writes into a fixed-size
output canvas; the AR baseline reads up to 512 tokens and writes until it stops. Inputs are
k consecutive test sentences joined into one, so the length grows in a controlled way and the
reference (the k reference sentences joined) stays exact. Prediction: NCM's BLEU and
output/reference length ratio collapse once inputs pass ~96 tokens; the AR model degrades
slowly.
"""

from .common import free_gpu, load_model, load_task, model_label, quality, words, write

GROUPS = (1, 2, 3, 4, 6, 8)


def join_groups(src: list[str], refs: list[str], k: int, n: int):
    """n inputs of k consecutive sentences each (fewer if the data runs out)."""
    stop = min(n * k, len(src) - len(src) % k)
    return ([" ".join(src[i : i + k]) for i in range(0, stop, k)],
            [" ".join(refs[i : i + k]) for i in range(0, stop, k)])


def input_tokens(model, texts: list[str]) -> list[int] | None:
    tok = getattr(model, "src_tok", None) or getattr(model, "tok", None)
    return [len(tok(t)["input_ids"]) for t in texts] if tok is not None else None


def evaluate(model, label: str, src: list[str], refs: list[str], groups=GROUPS,
             n: int = 100) -> tuple[list[dict], list[dict]]:
    rows, details = [], []
    limit = getattr(model, "prefix_len", None) or getattr(model, "max_input_tokens", None)
    for k in groups:
        inputs, targets = join_groups(src, refs, k, n)
        if not inputs:
            continue
        hyps = model.generate(inputs)
        n_tok = input_tokens(model, inputs)
        row = {"model": label, "sentences_per_input": k, "n": len(inputs),
               "input_words": words(inputs), "ref_words": words(targets),
               "output_words": words(hyps),
               "length_ratio": round(words(hyps) / max(words(targets), 1e-9), 3),
               **quality("wmt14", hyps, targets), "input_limit_tokens": limit}
        if n_tok is not None:
            row["input_tokens"] = round(sum(n_tok) / len(n_tok), 1)
            row["share_truncated"] = round(sum(t > limit for t in n_tok) / len(n_tok), 3) \
                if limit else None
        rows.append(row)
        details += [{"model": label, "k": k, "src": s, "ref": r, "hyp": h}
                    for s, r, h in list(zip(inputs, targets, hyps))[:5]]
    return rows, details


def main(n: int = 100, precision: str = "fp16") -> None:
    _, src, refs = load_task("wmt14", n * max(GROUPS))
    rows, details = [], []
    # the AR model gets room for long outputs: its usual 64-token cap would be the limit
    for kind, kw in [("ncm", {"batch_size": 32}),
                     ("ar", {"batch_size": 16, "max_new_tokens": 400})]:
        model = load_model(kind, "wmt14", precision, **kw)
        r, d = evaluate(model, model_label(kind, "wmt14"), src, refs, n=n)
        rows += r
        details += d
        del model
        free_gpu()
    write("input_length", rows, details)
