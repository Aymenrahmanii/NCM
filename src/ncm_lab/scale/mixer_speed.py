"""Phase 11 speed: batch-1 time and memory of one NCM sample per generator mixer and length.

    python -m ncm_lab.scale.mixer_speed --device cuda
    python -m ncm_lab.scale.mixer_speed --device cpu --repeats 5

Random weights at the gen25m width and depth (speed does not depend on what was learned);
the tokenizer's levels double from 1 up to the sample length, as in Phase 10. Mixers are
interleaved within each repeat so drift (heat, other processes) hits all of them alike.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import time
from pathlib import Path

import torch

from .models import MIXERS, Generator, Tokenizer, TokenizerConfig, ncm_sample
from .train import GENERATORS, TOKENIZER

OUT = Path("results")
LENGTHS = (256, 1024, 2048)


def tokenizer_config(length: int) -> TokenizerConfig:
    levels = tuple(2 ** i for i in range(length.bit_length()) if 2 ** i <= length)
    return dataclasses.replace(TOKENIZER, seq_len=length, levels=levels)


def bench(dev: str, repeats: int, lengths=LENGTHS, mixers=MIXERS, size: str = "gen25m",
          warmup: int = 3) -> list[dict]:
    rows = []
    for length in lengths:
        tcfg = tokenizer_config(length)
        tok = Tokenizer(tcfg).to(dev).eval()
        gens = {m: Generator(dataclasses.replace(GENERATORS[size], mixer=m), tcfg).to(dev).eval()
                for m in mixers}
        times: dict[str, list[float]] = {m: [] for m in mixers}
        peak: dict[str, float] = {}
        for r in range(warmup + repeats):
            for m, gen in gens.items():
                if dev == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=dev == "cuda"):
                    ncm_sample(gen, tok, 1, temperature=1.0)
                if dev == "cuda":
                    torch.cuda.synchronize()
                    peak[m] = torch.cuda.max_memory_allocated() / 2**20
                if r >= warmup:
                    times[m].append(time.perf_counter() - t0)
        for m in mixers:
            rows.append({"device": dev, "length": length, "mixer": m,
                         "median_ms": round(1000 * statistics.median(times[m]), 2),
                         "peak_mib": round(peak.get(m, float("nan")), 1),
                         "repeats": repeats})
            print(rows[-1], flush=True)
        del gens, tok
        if dev == "cuda":
            torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args(argv)
    rows = bench(a.device, a.repeats)
    with open(OUT / f"phase11_speed_{a.device}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
