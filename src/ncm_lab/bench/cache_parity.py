"""Cache bisect: does the KV-cache path of the official NCM generate() give the same codes?

    python -m ncm_lab.bench.cache_parity --n 8 --batch 1      # unpadded, one item at a time
    python -m ncm_lab.bench.cache_parity --n 8 --batch 8      # padded batches of 8

Decoding is greedy (top_k=1 in the checkpoint config), so cached and uncached runs must give
identical codes. The report says, per level, how many items diverge and where the first
divergence is.
"""

import argparse
from pathlib import Path

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
        "--layers", action="store_true", help="layer-level divergence, cached vs uncached"
    )
    ap.add_argument("--profile", action="store_true", help="profile uncached vs cached at --batch")
    ap.add_argument("--speed", action="store_true", help="time uncached vs cached+pad fix")
    ap.add_argument("--speed-reps", type=int, default=20)
    ap.add_argument(
        "--compile",
        default="none",
        choices=["none", "default", "reduce-overhead", "cudagraphs"],
        help="torch.compile the transformer blocks before --speed",
    )
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
    if args.layers:
        layer_divergence(model)
        free_gpu()
        return
    if args.profile:
        profile_both(model, args.batch)
        free_gpu()
        return
    if args.speed:
        install_pad_fix(type(model.net))
        if args.compile != "none":
            compile_blocks(model, args.compile)
        print(speed(model, args.batch, args.speed_reps))
        free_gpu()
        return
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
    tag = str(model.dtype).split(".")[-1]
    out_dir = Path("results/cache_parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, h in hyps.items():
        (out_dir / f"wmt14_{name.replace('+', '_')}_{tag}_n{n}.txt").write_text(
            "\n".join(h) + "\n", encoding="utf-8"
        )
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


def speed(model, batch: int, reps: int, warmup: int = 3) -> dict[str, float]:
    """Median ms per batch, uncached vs cached (with the pad fix), interleaved A,B,A,B.

    Caller installs the pad fix first. Each condition runs `warmup` untimed calls,
    then `reps` timed calls, with the GPU synchronized around the clock.
    """
    import statistics
    import time

    inputs, _, _ = load_task("wmt14", batch)
    prefixes = model.prepare(inputs[:batch])
    times: dict[str, list[float]] = {"uncached": [], "cached": []}
    for use_cache, key in ((False, "uncached"), (True, "cached")):
        for _ in range(warmup):
            translate(model, prefixes, use_cache)
    for _ in range(reps):
        for use_cache, key in ((False, "uncached"), (True, "cached")):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            translate(model, prefixes, use_cache)
            torch.cuda.synchronize()
            times[key].append((time.perf_counter() - t0) * 1000)
    out = {}
    for key, ts in times.items():
        out[f"{key}_median_ms"] = round(statistics.median(ts), 1)
    out["speedup"] = round(out["uncached_median_ms"] / out["cached_median_ms"], 2)
    out["batch"] = batch
    out["reps"] = reps
    return out


def profile_both(model, batch: int, top: int = 12) -> None:
    """Print the top ops by self CPU time for uncached and cached (pad fix) decoding."""
    from torch.profiler import ProfilerActivity, profile

    inputs, _, _ = load_task("wmt14", batch)
    prefixes = model.prepare(inputs[:batch])
    translate(model, prefixes, False)  # warm-up
    for use_cache in (False, True):
        if use_cache:
            install_pad_fix(type(model.net))
            translate(model, prefixes, True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            translate(model, prefixes, use_cache)
            torch.cuda.synchronize()
        label = "cached+pad_fix" if use_cache else "uncached"
        print(f"== {label}, batch {batch}")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=top))


def compile_blocks(model, mode: str) -> None:
    """Compile each transformer block's forward (test-only; both paths call the same blocks)."""
    for block in model.net.transformer.blocks:
        if mode == "cudagraphs":  # CUDA graphs only: no C compiler needed
            block.forward = torch.compile(block.forward, backend="cudagraphs")
        else:
            block.forward = torch.compile(block.forward, mode=mode)


def block_outputs(model, prefixes: torch.Tensor, use_kv_cache: bool) -> list[torch.Tensor]:
    """Every transformer block's output, in call order, as float tensors."""
    store: list[torch.Tensor] = []
    handles = [
        blk.register_forward_hook(lambda m, i, o: store.append(o.detach().float()))
        for blk in model.net.transformer.blocks
    ]
    try:
        codes_per_level(model, prefixes, use_kv_cache)
    finally:
        for h in handles:
            h.remove()
    return store


def layer_divergence(model, n: int = 16, fp_tol: float = 1e-2) -> None:
    """Find the first item whose codes differ (fp16), then the first layer where activations do."""
    inputs, _, _ = load_task("wmt14", n)
    prefixes = model.prepare(inputs[:n])
    eager_codes = codes_per_level(model, prefixes, False)
    install_pad_fix(type(model.net))
    cached_codes = codes_per_level(model, prefixes, True)
    differ = torch.zeros(n, dtype=torch.bool, device=prefixes.device)
    for a, b in zip(eager_codes, cached_codes):
        differ |= (a != b).any(dim=1)
    if not differ.any():
        print(f"no codes differ among the first {n} items")
        return
    item = int(differ.nonzero()[0])
    print(f"item {item} differs; {int(differ.sum())}/{n} items differ in this batch")

    eager = block_outputs(model, prefixes, False)
    cached = block_outputs(model, prefixes, True)
    n_blocks = len(model.net.transformer.blocks)
    n_levels = len(model.net.levels) - 1
    # the prelude passes differ between the paths; the level calls are the last ones
    eager, cached = eager[-n_levels * n_blocks :], cached[-n_levels * n_blocks :]
    P = model.net.prefix_len
    first = None
    for call, (e, c) in enumerate(zip(eager, cached)):
        lvl, blk = divmod(call, n_blocks)
        diff = (e[item, P:] - c[item]).abs().max().item()
        scale = e[item, P:].abs().max().item()
        if first is None and diff > fp_tol * max(scale, 1.0):
            first = (lvl + 1, blk, diff, scale)
    if first is None:
        print("activations agree within tolerance at every level and block")
    else:
        lvl, blk, diff, scale = first
        print(
            f"first divergence: level {lvl}, block {blk} (max abs diff {diff:.4f}, scale {scale:.2f})"
        )


if __name__ == "__main__":
    main()
