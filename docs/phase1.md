# Phase 1: Python and PyTorch foundations

## What a training loop does

A model is a function with millions of adjustable numbers (weights). One training step:

1. **Forward**: feed a batch of inputs through the model to get predictions.
2. **Loss**: one number saying how wrong the predictions are (cross-entropy for "which
   character comes next").
3. **Backward**: `loss.backward()` uses autograd to compute, for every weight, which direction
   to nudge it so the loss goes down (the gradient).
4. **Step**: the optimizer (AdamW) moves every weight a little in that direction.
   `zero_grad()` clears the old gradients first, because PyTorch adds them up by default.

Repeat thousands of times. In `scripts/char_model.py` the loss went from 4.37 (random guessing
among 65 characters: ln 65 ≈ 4.17) to about 1.55 on validation data.

Train vs validation loss: train is measured on text the model learns from, validation on
text it never sees. When train keeps dropping but validation stops (step ~2400 here), the
model starts memorising instead of generalising.

## How a normal LLM generates

`CharTransformer.generate` produces one character per forward pass: predict the next
character, append it, feed the longer text back in. Character 10 cannot be computed before
character 9 exists. 500 characters = 500 forward passes, one after another.

## Why NCM-style timing stays low as TOKENS grows

Result from `scripts/parallel_vs_sequential.py` (RTX 3050, `results/phase1_parallel_vs_sequential.txt`):

| tokens | sequential ms | parallel ms | ratio |
|---|---|---|---|
| 16 | 2.6 | 0.10 | 27× |
| 64 | 5.2 | 0.16 | 32× |
| 256 | 20.6 | 0.14 | 147× |
| 1024 | 77.3 | 0.25 | 313× |

Sequential work is a chain: each call waits for the previous one, so time grows linearly
with tokens, and each small call leaves most of the GPU idle. Parallel work is one big
matrix multiply: the GPU spreads all positions over its thousands of cores at once, so
the time barely changes until the GPU is full. NCM generates a whole level in one pass,
which is why its cost depends on the number of levels (~5), not the number of tokens.

Caveat: this toy ignores that a real parallel model must produce good text without seeing
its neighbours' choices. That quality cost is what Phase 2 and Phase 4 measure.

## GPU timing rules learned

- Warm up before timing (first CUDA calls pay one-off setup costs).
- `torch.cuda.synchronize()` before reading the clock: GPU calls return immediately and
  run asynchronously, so without it you only time how long it took to queue work.

## Environment notes

- WSL2 Ubuntu 24.04, Python 3.12, PyTorch 2.14.1+cu130, RTX 3050 6 GB.
- `python3-venv` needs sudo; `uv venv --seed .venv` works without it.
- Git in WSL reuses the Windows Git Credential Manager for GitHub login.
- Tags are named `phase-N-done` because a tag with the same name as a branch makes
  `git push` ambiguous.
