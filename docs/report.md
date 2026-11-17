# Final report: Next-Concept Models on a laptop GPU

> Draft written with Claude. Rewrite it in your own words.

This closes the research plan in `docs/research-plan.md` (Phases 7–12, 2026-11-11 to
2026-11-17). The full technical write-up is `paper/main.tex`; each phase has its own doc with
every table. This page gives the answers, the evidence for each, and what they do not show.

## The question

NCM (Hazy Research, 2026) writes text coarse to fine: one pass per level, all positions of a
level in parallel. It is reported to be up to 31× faster than a diffusion language model on an
H100 at 1,024 tokens. The plan asked what that speed is worth on a 6 GB RTX 3050 against the
baseline most practitioners use, an autoregressive (AR) model, and whether the method itself,
apart from who trained which checkpoint, beats AR at matched training.

## Answers

| goal | answer | evidence |
|---|---|---|
| 1. Is NCM faster than AR on this GPU? | **At batch 1, yes when both sides are compiled** (1.2–1.7×), a tie or a loss in eager mode, and **slower at batch 32** (1.7–2.0×). On the CPU it is 1.1–1.7× slower. | Phase 8, `docs/phase4-results.md` |
| 1b. At long outputs? | 6–10× faster than GPT-2 small at batch 1 from 128 to 1,024 tokens, but the lead does not grow with length and is gone by batch 8 at 512. | Phase 9, `docs/phase9.md` |
| 2. Does the method beat the AR frontier at matched training? | **No.** Same data, BPE and 200M positions: 35M NCM MAUVE 0.006 vs 0.59–0.92 for AR models of 1.9–35.9M (human 0.95), at about 20× lower latency. Three seeds, seven decoding settings and a blind LLM judge agree. | Phase 10, `docs/phase10.md` |
| 2b. Is that a weak reimplementation? | **Not mainly.** An official-style level schedule makes the finest level nearly perfect (0.94 given the coarser levels) but free samples still score 0.007; the generator cannot write the coarse levels at this scale. | Phase 10 validity check |
| 3. Does the generator need full attention? | **It is not the bottleneck.** All mixers score the same floor; only convolution is faster (1.5× on the CPU at 1,024), with worse training loss. | Phase 11, `docs/phase11.md` |
| 4. How cheaply can a released NCM be adapted? | **Rank-16 LoRA on every layer**: within 0.9 ROUGE-L of full fine-tuning on SAMSum with under 3% of the weights (BART: 0.4). The common rank-8 query/value default is 3–4 points short. | Phase 12, `docs/phase12.md` |

## The strongest honest claims

These are the plan's pre-registered claim templates, filled in with what was measured:

- **Phase 10 (the main result):** the closest template is *"NCM's speed costs quality at this
  scale"*, but its first half ("a smaller AR model is both faster and more fluent") does not
  hold: no AR model here is as fast. What was measured: at 25M parameters and 200M training
  positions on TinyStories, NCM is about 20× faster per sample than the AR ladder and two
  orders of magnitude worse on MAUVE; even the 1.9M AR model, 11× slower, is far more fluent.
- **Phase 8:** *the released checkpoints' batch-32 throughput deficit is largely an
  implementation gap.* With the corrected KV cache (same fix as upstream PR #5), NCM goes
  from 2.6× to level with opus-mt on WMT14 and from 9.8× to 2.2× behind BART on XSum, at
  unchanged quality.
- **GPU vs CPU:** *NCM's advantage is unused GPU parallelism at batch 1; it does not carry
  over to CPU-only devices.*
- **Phase 12:** *a released NCM adapts to a new summarization domain with under 3% trainable
  parameters* (the plan's "under 1%" template is not met: the 0.3% arm is 3.7 points short).
- **Not supported:** the attention-replacement claim. No mixer is removable "at no
  measurable cost" by the only usable signal, training loss.

## What changed along the way

- **The 31× is against diffusion, not AR.** Against same-size AR models the official
  checkpoints roughly tie at batch 1 on this GPU.
- **Two measurement bugs moved results.** The official KV cache had a padding bug (cached
  outputs differed from uncached); fixed, it changes throughput by 2.5–4.5×. A torch.compile
  recompile limit silently fell back to eager in a first compiled run; it is now guarded.
- **The finest-level story was revised.** Phase 10 first located the failure at the last
  level. The validity check showed that this was partly our doubling level schedule; with the
  official-style schedule the failure moves to the coarse levels. The stop verdict did not
  change, its reason did.
- **The 40-question RAG sample overstated every accuracy** (plain RAG 87.5% → 73.3% on all
  1,190 XQuAD questions).

## Limits of these results

- One laptop GPU (RTX 3050, 6 GB, thermally limited at 87 °C); absolute times vary between
  runs, so ratios come from interleaved timing.
- The controlled comparison is toy scale: 25–35M parameters, 256-token stories, 200M
  positions. The released 125M TinyStories NCM reaches MAUVE 0.80, so NCM may close the gap
  at larger scale; this study cannot say.
- The validity check and Phase 11 use one seed at half the main budget; the validity
  generator changes the level schedule and tied heads together.
- Pretrained baselines (BART, opus-mt) against task-trained NCM checkpoints in Phases 4, 5
  and 12, so those quality gaps are not purely the method's.
- Faithfulness is measured with string proxies; the LLM judge is a 3B model that cannot rank
  the AR sizes.

## What would be worth doing next

- **Scale the controlled comparison** to the released model's size (125M) on a larger GPU, the
  one experiment that could change the Phase 10 verdict.
- **Give the coarse levels more help:** a small AR head on the first levels, or iterative
  refinement, since the generator refines well and plans badly.
- **Contribute to upstream PR #5** with the independent RTX 3050 run (same bug, same fix).

## Reproducing

`bash scripts/reproduce_results.sh` recreates every number; individual stages take a phase
name (`phase8` … `phase12`, plus `phase10-train` and `phase11-train`). Training and evaluation
commands are listed in each phase doc. The Phase 10 training chain alone is about 24
GPU-hours on the RTX 3050.
