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
