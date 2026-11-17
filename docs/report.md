# Final report: Next-Concept Models on a laptop GPU

This report closes the research plan in `docs/research-plan.md` (Phases 7–12, run between
2026-11-11 and 2026-11-17). The full technical write-up is `paper/main.tex`, and each phase
has its own doc with all the tables. Here I give the answers, where the evidence is, and
what the results don't show.

## The question

NCM (Hazy Research, 2026) writes text coarse to fine: one pass per level, with every position
of a level predicted in parallel. It's reported to be up to 31× faster than a diffusion
language model, measured on an H100 at 1,024 tokens. I wanted to know what that speed is worth
on a 6 GB RTX 3050 against the baseline most people actually use, an autoregressive (AR)
model. And beyond who trained which checkpoint, does the method itself beat AR when both are
trained the same way?

## Answers

| goal | answer | evidence |
|---|---|---|
| 1. Is NCM faster than AR on this GPU? | At batch 1, yes when both sides are compiled (1.2–1.7×); without compilation it ties or loses. At batch 32 it's 1.7–2.0× slower, and on the CPU 1.1–1.7× slower. | Phase 8, `docs/phase4-results.md` |
| 1b. And at long outputs? | 6–10× faster than GPT-2 small at batch 1 from 128 to 1,024 tokens. The lead doesn't grow with length, and it's gone by batch 8 at 512 tokens. | Phase 9, `docs/phase9.md` |
| 2. Does the method beat the AR frontier with matched training? | No. Same data, BPE and 200M positions: the 35M NCM gets MAUVE 0.006, against 0.59–0.92 for AR models of 1.9–35.9M (human text: 0.95), while being about 20× faster. Three seeds, seven decoding settings and a blind LLM judge all agree. | Phase 10, `docs/phase10.md` |
| 2b. Is that just a weak reimplementation? | Mostly not. With an official-style level schedule the finest level becomes nearly perfect (0.94 when the coarser levels are given), but free samples still score 0.007. At this scale the generator can't write the coarse levels. | Phase 10, validity check |
| 3. Does the generator need full attention? | Attention isn't the bottleneck. Every mixer lands on the same quality floor; only convolution is faster (1.5× on the CPU at 1,024 positions), and its training loss is worse. | Phase 11, `docs/phase11.md` |
| 4. How cheaply can a released NCM be adapted? | Rank-16 LoRA on every layer comes within 0.9 ROUGE-L of full fine-tuning on SAMSum, training under 3% of the weights (BART: 0.4). The usual rank-8 query/value default is 3–4 points short. | Phase 12, `docs/phase12.md` |

## The claims I can stand behind

Before running anything, the plan listed the claim I'd be allowed to make for each possible
outcome. Here is how they turned out.

- Phase 10, the main result. The closest template was "NCM's speed costs quality at this
  scale", but its first half, "a smaller AR model is both faster and more fluent", doesn't
  hold: none of the AR models is as fast. What I measured is this. At 25M parameters and 200M
  training positions on TinyStories, NCM is about 20× faster per sample than the AR ladder
  and two orders of magnitude worse on MAUVE. Even the 1.9M AR model, 11× slower, is far more
  fluent.
- Phase 8. The released checkpoints' batch-32 throughput gap is largely an implementation
  problem. With the corrected KV cache (the same fix as upstream PR #5), NCM goes from 2.6×
  behind opus-mt on WMT14 to level with it, and from 9.8× to 2.2× behind BART on XSum, with
  unchanged quality.
- GPU vs CPU. NCM's advantage comes from GPU parallelism that sits unused at batch 1, and it
  doesn't carry over to CPU-only machines.
- Phase 12. A released NCM can be adapted to a new summarization domain while training under
  3% of its parameters. The plan's template said "under 1%", and that one fails: the 0.3% arm
  is 3.7 points short.
- What I can't claim: that attention can be replaced. Training loss, the only signal that
  separates the mixers, says no mixer comes "at no measurable cost".

## What changed along the way

- The 31× turned out to be against diffusion, not AR. Against same-size AR models the
  official checkpoints roughly tie at batch 1 on this GPU.
- Two measurement bugs moved the results. The official KV cache had a padding bug, so cached
  outputs didn't match uncached ones; fixing it changes throughput by 2.5–4.5×. In a first
  compiled run, torch.compile hit its recompile limit and silently fell back to eager mode;
  that's now guarded against.
- I had to revise where Phase 10 fails. At first I located the failure at the last level.
  The validity check showed that this was partly caused by my own doubling level schedule;
  with an official-style schedule the failure moves to the coarse levels. The verdict to stop
  stayed the same, but the reason changed.
- The 40-question RAG sample had overstated every accuracy (plain RAG went from 87.5% to
  73.3% on all 1,190 XQuAD questions).

## Limits

- Everything ran on one laptop GPU (RTX 3050, 6 GB, held at 87 °C by its thermal limit).
  Absolute times move between runs, which is why the ratios come from interleaved timing.
- The controlled comparison is small: 25–35M parameters, 256-token stories, 200M positions.
  The released 125M TinyStories NCM reaches MAUVE 0.80, so NCM may close the gap at larger
  scale. This study can't tell.
- The validity check and Phase 11 use one seed at half the main training budget, and the
  validity generator changes the level schedule and the tied heads at the same time.
- In Phases 4, 5 and 12 the baselines (BART, opus-mt) are pretrained while the NCM checkpoints
  were trained only on their task, so those quality gaps aren't purely down to the method.
- Faithfulness is measured with string-matching proxies, and the LLM judge is a 3B model that
  can't tell the AR sizes apart.

## What I'd do next

- Scale the controlled comparison up to the released model's size (125M) on a bigger GPU.
  It's the one experiment that could overturn the Phase 10 verdict.
- Help the coarse levels, for example with a small AR head on the first levels or with
  iterative refinement. The generator refines well; it plans badly.
- Add my independent RTX 3050 run to upstream PR #5 (same bug, same fix).

## Reproducing

`bash scripts/reproduce_results.sh` rebuilds every number. Individual stages take a phase
name (`phase8` … `phase12`, plus `phase10-train` and `phase11-train`), and each phase doc lists
its training and evaluation commands. The Phase 10 training chain on its own takes about 24
GPU-hours on the RTX 3050.

---

*Written with help from Claude (Anthropic). Every number comes from the files in `results/`.*
