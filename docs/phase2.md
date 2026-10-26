# Phase 2: a toy NCM from scratch

> Draft written with Claude. Rewrite it in your own words: explaining it yourself is the test.

Code: `src/ncm_lab/toy_ncm/`. Results: `results/phase2_*`.

## The idea in one paragraph

A normal language model writes left to right, one token per forward pass. NCM writes
**coarse to fine**: first a tiny "gist" of the whole text, then a slightly longer sketch
that refines it, and so on, until the last level has one entry per token. Every position
inside a level is predicted **at the same time**, so the number of forward passes equals the
number of levels (6 here), not the number of tokens (64 here).

## The two stages

```
 Stage 1: multiscale tokenizer (trained first, then frozen)

 64 tokens ─► encoder ─► z (64 × 32) ─┬─ level 1: average to 1 position  ─► nearest code ─► stretch back to 64 ─┐
                                      │  residual = z − sum so far                                            │ sum
                                      ├─ level 2: average to 4 positions ─► nearest code ─► stretch ──────────┤ = canvas
                                      ├─ ...                                                                    │
                                      └─ level 6: 64 positions           ─► nearest code ─► (already 64) ─────┘
                                                                                  canvas ─► decoder ─► 64 tokens

 Stage 2: generator (tokenizer frozen)

 level 1 input: empty canvas, shrunk to 1     ─┐
 level 2 input: canvas after level 1, shrunk to 4 ─┤   one transformer, block-causal mask:
 ...                                            ├─► level k sees levels 1..k only ─► codes of every level
 level 6 input: canvas after level 5 (64)      ─┘
```

**Vector quantization.** Each level has its own codebook of 512 learned vectors. A vector is
replaced by its nearest code. Picking "the nearest" has no gradient, so training uses the
**straight-through trick**: forward uses the code, backward pretends the snap never happened
(`q = r + (q - r).detach()`). A **commitment loss** keeps the encoder's vectors close to
the codes, and a codebook loss pulls the codes towards the data.

**Residual levels.** Level 1 quantizes a 1-position average of the whole text. Level 2
quantizes what level 1 got wrong, at 4 positions, and so on. Later levels only add detail.

**Nested dropout.** During training, half the batch is decoded from only the first k levels
(k random). That forces the early levels to carry the most important information alone.

**Generator.** For each level it sees the canvas built so far (shrunk to the level's length)
and predicts all codes of that level at once. All 6 levels train in one forward pass using
a block-causal attention mask. Up to 25% of the conditioning codes are replaced by random
ones during training, so the model learns to cope with its own sampling mistakes.

## Results (RTX 3050)

Sizes: tokenizer 8.6M, generator 5.6M, AR baseline 6.9M. Trained on the first 64 BPE tokens
of 300k TinyStories, vocabulary 4096.

**Reconstruction accuracy rises level by level** (`results/phase2_recon_by_level.png`):

| levels used | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| accuracy | 17.0% | 20.7% | 28.2% | 45.1% | 73.1% | 98.7% |

Level 1 is a single code for the whole story, and it already decodes to
"Once upon a time, there was a little…". Full printouts: `results/phase2_level_by_level.txt`.

**Speed** (64 tokens, bf16, generation only):

| batch | toy NCM | AR + KV cache | NCM speedup |
|---|---|---|---|
| 1 | 34 ms | 316 ms | 9.3× |
| 16 | 37 ms | 287 ms | 7.7× |
| 128 | 89 ms | 269 ms | 3.0× |

**Quality** (GPT-2 perplexity of 512 samples, lower is more fluent): real stories 13.2,
AR 24.5, toy NCM 377. NCM samples start well ("Lily and Max were playing hide and seek in
the park") and fall apart after a sentence or two.

## What I learned

- The speed advantage is real at small batch and shrinks at large batch: the AR model's
  64 small passes waste the GPU at batch 1, but at batch 128 each pass is already big.
- Parallel prediction has a quality cost. Positions inside a level are sampled
  independently, so neighbouring words can disagree ("hide and seek and seek").
  Small models suffer most; the paper's ~100M models do far better (Phase 3).
- My first tokenizer run had a VQ loss that kept growing (encoder vectors drifting away
  from the codebooks). A LayerNorm before quantizing fixed it; the official config does
  the same with `pre_quant_groupnorm`.
- Dead codes (never chosen) waste capacity; restarting them on recent data vectors keeps
  the codebooks in use.
