"""Cache bisect: does the KV-cache path of the official NCM generate() give the same codes?

    python -m ncm_lab.bench.cache_parity --n 8 --batch 1      # unpadded, one item at a time
    python -m ncm_lab.bench.cache_parity --n 8 --batch 8      # padded batches of 8

Decoding is greedy (top_k=1 in the checkpoint config), so cached and uncached runs must give
identical codes. The report says, per level, how many items diverge and where the first
divergence is.
"""

import argparse

import torch

from ncm_lab.limits.common import free_gpu, load_model, load_task


PAD_FIX_EDITS = [
    # the cached prefix pass must mask pad tokens like the uncached path does
    (
        "                pfx_mask = torch.zeros(1, 1, P, P, device=prefix_emb.device)\n",
        "                pfx_mask = torch.zeros(1, 1, P, P, device=prefix_emb.device)\n"
        "                if _pbias is not None:\n"
        "                    pfx_mask = pfx_mask + _pbias\n",
    ),
    # and so must every level's query block over the cached prefix
    (
        "                    lvl_mask = torch.zeros(1, 1, L_cur, P + L_cur, device=x.device)\n",
        "                    lvl_mask = torch.zeros(1, 1, L_cur, P + L_cur, device=x.device)\n"
        "                    if _pbias is not None:\n"
        "                        lvl_mask = lvl_mask + F.pad(_pbias, (0, L_cur))\n",
    ),
]


def install_pad_fix(model_cls) -> None:
    """Replace NextConceptTransformer.generate with a copy that masks prefix pads in the cache path.

    Test-only: the third_party clone stays unchanged. Each edit must match exactly once.
    """
    import inspect
    import sys
    import textwrap

    module = sys.modules[model_cls.__module__]
    src = inspect.getsource(model_cls.generate)  # anchors use the file's indentation
    for old, new in PAD_FIX_EDITS:
        if src.count(old) != 1:
            raise RuntimeError(f"pad-fix anchor matched {src.count(old)} times")
        src = src.replace(old, new)
    src = textwrap.dedent(src)
    namespace = dict(vars(module))
    exec(compile(src, "<pad-fix generate>", "exec"), namespace)
    model_cls.generate = namespace["generate"]


@torch.no_grad()
def codes_per_level(model, prefix: torch.Tensor, use_kv_cache: bool) -> list[torch.Tensor]:
    """Generated codes split by level, shape (B, level_size) each."""
    with model._autocast():
        codes = model.net.generate(
            None,
            prefix,
            temperature=model.temperature,
            top_k=model.top_k,
            top_p=model.top_p,
            use_kv_cache=use_kv_cache,
        )
    sizes = list(model.net.levels)
    if sum(sizes) != codes.shape[1]:
        raise ValueError(f"levels {sizes} do not add up to {codes.shape[1]} codes")
    out, start = [], 0
    for s in sizes:
        out.append(codes[:, start : start + s])
        start += s
    return out


def compare(model, prefix: torch.Tensor) -> list[dict]:
    """One row per level: items that diverge, and the first differing position."""
    eager = codes_per_level(model, prefix, use_kv_cache=False)
    cached = codes_per_level(model, prefix, use_kv_cache=True)
    rows = []
    for k, (a, b) in enumerate(zip(eager, cached)):
        diff = a != b
        bad_items = diff.any(dim=1)
        first = None
        if bad_items.any():
            item = int(bad_items.nonzero()[0])
            pos = int(diff[item].nonzero()[0])
            first = {
                "item": item,
                "pos": pos,
                "eager": int(a[item, pos]),
                "cached": int(b[item, pos]),
            }
        rows.append(
            {
                "level": k,
                "size": a.shape[1],
                "items": a.shape[0],
                "items_diverging": int(bad_items.sum()),
                "first": first,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=8, help="number of WMT14 sentences")
    ap.add_argument("--batch", type=int, default=1, help="items per forward pass")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument(
        "--bleu",
        type=int,
        default=0,
        help="corpus BLEU on this many sentences: uncached, cached, cached+pad fix",
    )
    ap.add_argument(
        "--pad-fix",
        action="store_true",
        help="test the proposed fix: mask prefix pads in the cached path",
    )
    args = ap.parse_args()

    model = load_model("ncm", "wmt14", args.precision)
    if args.bleu:
        print(bleu_report(model, args.bleu))
        free_gpu()
        return
    if args.pad_fix:
        install_pad_fix(type(model.net))
    inputs, _, _ = load_task("wmt14", args.n)
    prefixes = model.prepare(inputs)

    for start in range(0, len(prefixes), args.batch):
        chunk = prefixes[start : start + args.batch]
        print(f"== batch of {chunk.shape[0]} starting at item {start}")
        for row in compare(model, chunk):
            print(
                f"  level {row['level']} (size {row['size']}): "
                f"{row['items_diverging']}/{row['items']} items diverge, first={row['first']}"
            )
    free_gpu()


@torch.no_grad()
def translate(model, prefixes: torch.Tensor, use_kv_cache: bool) -> list[str]:
    """Same decoding as NCMModel.run, with the cache flag passed through."""
    with model._autocast():
        codes = model.net.generate(
            None,
            prefixes,
            temperature=model.temperature,
            top_k=model.top_k,
            top_p=model.top_p,
            cfg_scale=model.cfg_scale,
            cfg_null_eos=model.eos,
            use_kv_cache=use_kv_cache,
        )
        _, z_levels = model.tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        tokens = model._pick_tokens(model.tk.vqvae.decoder(z_levels[-1]).float())
    return model.finish(tokens)


def bleu_report(model, n: int) -> dict[str, object]:
    """Corpus BLEU on n WMT14 test sentences: uncached, cached, and cached with the pad fix.

    The pad fix patches the class, so the unfixed cached run comes first.
    """
    import sacrebleu

    inputs, _, refs = load_task("wmt14", n)
    prefixes = model.prepare(inputs)
    chunks = [prefixes[i : i + model.batch_size] for i in range(0, len(prefixes), model.batch_size)]

    def run_all(use_cache: bool) -> list[str]:
        out: list[str] = []
        for chunk in chunks:
            out += translate(model, chunk, use_cache)
        return out

    results: dict[str, object] = {}
    hyps = {}
    hyps["uncached"] = run_all(False)
    hyps["cached"] = run_all(True)
    install_pad_fix(type(model.net))
    hyps["cached+pad_fix"] = run_all(True)
    for name, h in hyps.items():
        results[name] = round(sacrebleu.corpus_bleu(h, [refs]).score, 2)
    results["identical_outputs_fixed_vs_uncached"] = sum(
        a == b for a, b in zip(hyps["uncached"], hyps["cached+pad_fix"])
    )
    results["identical_outputs_raw_vs_uncached"] = sum(
        a == b for a, b in zip(hyps["uncached"], hyps["cached"])
    )
    results["n"] = len(refs)
    return results


if __name__ == "__main__":
    main()
