# Phase 10: controlled comparison and scale-up

> Draft written with Claude. Rewrite it in your own words.

The plan is in `docs/research-plan.md`. This file collects results as they come in.

## Diagnostic: which level breaks coherence? (2026-11-14, no training)

```bash
python -m ncm_lab.toy_ncm.diagnose     # ~5 min on the RTX 3050, ~1 GB
```

The Phase 2 toy NCM (tokenizer 8.6M, generator 5.6M, 64 tokens, levels 1/4/8/16/32/64)
reconstructs 98.7% of tokens, yet its samples have GPT-2 perplexity 372 against 13.5 for real
stories. So the damage happens in the generator. For 512 test stories, the real codes of levels
1..k are kept and the generator writes the rest; the result is decoded and judged by GPT-2
small, as in Phase 2. Rows use fixed seeds and reproduce exactly.

**Hypothesis (from the plan):** the cause is independent sampling of every position within a
level, not capacity, so more parameters alone will help only partly.

### Where it breaks

| real levels kept | levels generated | GPT-2 ppl (sampled, T=1) | greedy | token agreement (sampled) |
|---|---|---|---|---|
| 1..6 | 0 (reconstruction) | 15.5 | – | 99.0% |
| 1..5 | 1 (the 64-code level) | **111.3** | 57.8 | 67.7% |
| 1..4 | 2 | 184.2 | 74.4 | 42.6% |
| 1..3 | 3 | 231.0 | 69.2 | 27.3% |
| 1..2 | 4 | 248.6 | 62.8 | 20.1% |
| 1..1 | 5 | 285.3 | 52.7 | 14.8% |
| none | 6 (free sample) | 372.3 | 67.8 | 7.5% |

Generating **only the finest level**, with every coarser level real, multiplies perplexity by
7.2 (15.5 → 111). Each earlier level adds a factor between 1.1 and 1.7. Coherence breaks at
the last level, where 64 codes, one per token, are drawn in parallel.

Teacher-forced, the generator's top code matches the real one only 36% of the time at that
level (loss 3.12 nats per code), even with all coarser levels real:

| level | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| positions | 1 | 4 | 8 | 16 | 32 | 64 |
| top-1 code accuracy | 0.0% | 9.1% | 10.7% | 14.9% | 21.6% | 35.8% |
| loss (nats/code; uniform = 6.24) | 6.07 | 5.30 | 5.04 | 4.54 | 3.98 | 3.12 |

### How much decoding fixes

| setting | GPT-2 ppl | distinct-2 (real: 0.423) |
|---|---|---|
| real 1..5, last level T = 1 / 0.7 / 0.5 / 0.3 / greedy | 111 / 80 / 67 / 61 / 58 | 0.477 / 0.427 / 0.394 / 0.374 / 0.365 |
| free sample, all levels T = 1 | 372 | 0.594 |
| free sample, last level T = 0.7 / 0.5 / 0.3 | 262 / 207 / 181 | 0.526 / 0.471 / 0.436 |
| free sample, all levels T = 0.7 | **144** | 0.437 |
| free sample, all levels greedy | 68 | 0.002 (one story repeated) |

- Even greedy decoding of the last level, given every real coarser level, stays at 58,
  4× the reconstruction's 15.5. In log terms, decoding closes about a third of the gap the
  last level opens (ln 111/15.5 = 1.97; ln 58/15.5 = 1.32).
- Lowering every level's temperature to 0.7 cuts free-sample perplexity 2.6× (372 → 144) and
  brings distinct-2 to the real stories' level (0.437 vs 0.423), so it is a better default than
  T = 1, not a collapse. Fully greedy sampling collapses (distinct-2 0.002).

### Verdict

- **Where:** the finest level. This is the one the plan's ablations already target (two
  refinement passes on the last two levels, per-level temperature).
- **Why:** the hypothesis holds only partly. Independent sampling explains about a third of
  the damage at the finest level; the rest is that the generator does not know the right
  fine code (36% top-1 with perfect context). That is a capacity and data problem: Phase 2
  trained the 5.6M generator on about 19M tokens. **Scaling data and parameters, as Phase 10
  plans, is necessary, not optional.**
- **For the Phase 10 runs:** tune temperature per level on the validation split (0.7 is a good
  starting point here); keep greedy out, it collapses; report the finest-level teacher-forced
  accuracy as a cheap progress metric during training.

### Caveats

- One toy model, one seed per row, 512 stories, temperatures picked on the test split (this
  is a diagnostic; Phase 10 tunes on validation).
- GPT-2 small judges 64-token prefixes of stories; perplexity rewards repetition, hence
  distinct-2 next to it.
- Level 1 has a single position: its 0% top-1 accuracy and near-uniform loss mean the
  generator spreads probability over many "gist" codes, which is expected for an
  unconditional first level and not a failure by itself.

Files: `results/phase10_diag_summary.md`, `_levels.csv`, `_teacher_forced.csv`,
`_samples.txt` (three stories completed every way).

## Scale-up setup (2026-11-14)

```bash
python -m ncm_lab.scale.data                 # full TinyStories, 256 tokens, ~8 min CPU
bash scripts/phase10_train.sh                # tokenizer, 2 generators, 5 AR models
bash scripts/phase10_train.sh seeds          # seeds 1 and 2 of gen25m and ar25m
python -m ncm_lab.scale.evaluate quality     # T tuned on val by MAUVE, scored on test
python -m ncm_lab.scale.evaluate latency     # batch 1, GPU (add --device cpu)
python -m ncm_lab.scale.evaluate plot
```

- **Data:** 2,119,489 training stories, 424M non-pad tokens, cut to 256 tokens with the
  Phase 2 BPE (4096). Only 21% of stories fill all 256 positions, so a budget of 200M
  positions is about 160M real tokens. The official validation split is halved into `val`
  (10,995 stories, temperature tuning) and `test` (10,995).
- **Models** (code in `src/ncm_lab/scale/`, same design as the toy NCM):

  | model | width × layers | params |
  |---|---|---|
  | tokenizer, levels 1, 2, 4, …, 256, 512 codes each | 288 × (4 enc + 4 dec) | 10.7M |
  | gen12m / gen25m | 384 × 6 / 512 × 7 | 12.5M / 24.6M |
  | ar2m / ar3m / ar7m / ar25m / ar35m | 128×4 / 192×4 / 256×6 / 512×7 / 512×10 | 1.9 / 3.4 / 6.9 / 26.4 / 35.9M |

  ar35m matches the NCM's total footprint (tokenizer + gen25m = 35.3M).
- **Held constant:** AdamW (0.9, 0.95, wd 0.01), lr 5e-4, cosine with 300 warm-up steps,
  bf16, batch 64 × 256, clip 1.0. Tokenizer 100M positions, everything else 200M.
- **Resumable:** checkpoints every 250 steps; batches are seeded by (seed, step), so a
  stopped and resumed run is bit-identical to an uninterrupted one (tested).
- **Measured cost on the RTX 3050**, each model alone (two in one process spill past 6 GB):
  tokenizer ~0.5 h, gen12m 1.5 h, gen25m 2.4 h (micro-batch 32), AR ladder 9 h in total.
  About 24 GPU-hours with the extra seeds, against the plan's estimate of 55.

### Tokenizer result (100M positions, 32 min)

Token reconstruction on 1024 val stories when decoding from the first k levels:

| levels used | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| finest length | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| accuracy | 10.1% | 10.3% | 10.6% | 11.0% | 12.4% | 17.2% | 30.6% | 58.3% | **96.6%** |

It passes the 95% gate. As in the toy (45% at a quarter of the length, 73% at half), the
coarse levels carry little token-level detail: at a quarter and half of the length this one
gets 31% and 58%. The text is mostly written by the last two levels, the ones the
diagnostic found fragile. Log: `artifacts/scale/tokenizer/log.csv`.
