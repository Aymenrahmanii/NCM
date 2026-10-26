"""Why parallel (NCM-style) generation stays fast as the sequence grows.

Sequential: one layer call per token, each output feeds the next step (like an LLM).
Parallel:   one layer call on all positions at once (like one NCM level).
"""

import time

import torch
import torch.nn as nn

DIM = 512
TOKENS_LIST = [16, 64, 256, 1024]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def time_it(fn, repeats: int = 5, warmup: int = 3) -> float:
    """Average seconds per call, measured after warm-up and with GPU sync."""
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - start) / repeats


def sequential(layer: nn.Module, tokens: int) -> torch.Tensor:
    x = torch.randn(1, DIM, device=DEVICE)
    for _ in range(tokens):
        x = torch.tanh(layer(x))  # step t needs the output of step t-1
    return x


def parallel(layer: nn.Module, tokens: int) -> torch.Tensor:
    x = torch.randn(tokens, DIM, device=DEVICE)
    return torch.tanh(layer(x))  # every position in one call


def main() -> None:
    torch.manual_seed(0)
    layer = nn.Linear(DIM, DIM).to(DEVICE)
    print(f"device: {DEVICE}")
    print(f"{'tokens':>7} {'sequential ms':>14} {'parallel ms':>12} {'ratio':>8}")
    with torch.no_grad():
        for tokens in TOKENS_LIST:
            seq = time_it(lambda t=tokens: sequential(layer, t)) * 1000
            par = time_it(lambda t=tokens: parallel(layer, t)) * 1000
            print(f"{tokens:>7} {seq:>14.3f} {par:>12.3f} {seq / par:>7.1f}x")


if __name__ == "__main__":
    main()
