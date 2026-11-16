# NCM research plan (continues from Phase 7)

> Draft written with Claude on 2026-11-11. Edit the assumptions and predictions into your own
> before running anything.

## Summary

The current results cannot yet separate "NCM the method" from "pretraining", "framework
overhead" and "missing caching", so the first three weeks repair the measurements and the
rest builds one fully controlled comparison. The single most important experiment is the
Phase 10 Pareto comparison. It trains NCM and a ladder of autoregressive (AR) models from
scratch on the same data, tokenizer and token budget. It then plots quality against batch-1
latency, with both sides optimized as hard as possible. The question is whether NCM beats the
AR *frontier* (including smaller, faster AR models), not one same-size AR model. Goals 2 and
3 only mean something once that frontier exists.

## Status (updated 2026-11-15)

| Item | State |
|---|---|
| Fix 1, limits suite | **Done** (commits `0f4de0a`, `952c824`). Tables are in `paper/tables/`; the paper's "TODO: interpretation" paragraphs are still open. `results/limits_run.log` is the stale log from the failed first run. |
| Fix 2, full test sets | **Done** for WMT14 and XSum (`bb420d0`) and for RAG (all 1,190 XQuAD questions, 2026-11-12). The 40-question sample had overestimated every RAG accuracy (plain RAG 87.5% → 73.3%). |
| Fixes 7–8, RAG scoring and German splitter | **Done** in the same rerun: exact match and token F1 next to containment; 1,265 German sentences with the fixed splitter (was 1,372). NCM vs AR is significant on every score in both experiments, including compression, which was not significant at n=40. |
| Statistics harness | **Done.** `ncm_lab/stats.py` (paired bootstrap for BLEU and mean metrics, Wilson, exact McNemar) and `ncm_lab.bench.significance`. All Phase 4 quality gaps are significant (see `results/significance.md`). |
| Fix 3 / Phase 8, cache bisect | **Done.** Padding bug; with the pad fix, cached fp16 is within 0.13 BLEU of uncached and fp32 matches 500/500. 1.87× at batch 32, 0.86× at batch 1. |
| Phase 8, upstream report | **Not needed as a new issue.** Upstream PR #5 (velaia, open since 2026-10-17, unreviewed) finds the same bug (missing `_pbias` in the cached prefix and level masks) and fixes it with the same edits. Contribute an independent RTX 3050 run as a comment on it instead. |
| Phase 8, XSum prefix cache | **Covered by PR #5**: it reuses the level-0 pass's prefix K/V, allows the cache for `query_level0` (XSum), and adds `bench_kv.py` and `eval_shard.py --kv-cache`. **Benchmarked here (2026-11-12):** 100% identical outputs; 2.46× on WMT14 and 4.48× on XSum at batch 16, 0.99× at batch 1; XSum peak memory goes from 2.0 to 4.9 GB (see `docs/phase4-results.md`). **Phase 4 rerun with PR #5 + #4, fp16, full test sets:** quality unchanged (bootstrap Δ includes 0); batch-32 throughput WMT14 19.1 → 49.5 ex/s (level with opus-mt, was 2.6× behind), XSum 5.2 → 23.1 ex/s (2.2× behind BART, was 9.8×); memory lower than uncached. Phase 8 success criterion (cached within 0.1 BLEU, batch-32 gap re-measured) is **met**. Remaining for Phase 8: compiled AR, CPU run. |
| Phase 8, interleaved timing | **Done (2026-11-12).** `ncm_lab.bench.interleaved`, full protocol (20 warm-up, 200 rotating rounds, paired-bootstrap ratio CIs, GPU temperature logged). NCM ÷ AR median latency at batch 1: WMT14 1.26 [1.14, 1.37] (**not a tie**; opus-mt is faster), XSum 0.99 [0.94, 1.05]. With the cache: WMT14 0.57 at batch 8, 1.03 at batch 32; XSum 0.97 at batch 8, 2.04 at batch 32. NCM's p95 is lower at batch 1 on both. See `docs/phase4-results.md`. |
| Phase 8, penalty loop | **Done (2026-11-13).** Incremental `pick_tokens`: identical picks (tests + all 11,334 XSum outputs), 120 → 63 ms and 640 → 62 MB scratch at batch 32. Only ~4% of a cached XSum batch, so not a bottleneck any more. Codes-to-logits not re-profiled yet. |
| Phase 8, compiled AR, CPU run | **Done (2026-11-13).** Compiled AR (static cache) keeps quality and is 2.2–2.3× faster than eager at batch 1. Interleaved, NCM (cache, fp16 weights, *not compiled*) ÷ compiled AR: 2.9× / 2.2× at batch 1, 1.5–1.7× at 8, 2.2× at 32 (WMT14 / XSum): NCM is slower everywhere. CPU, 4 threads, batch 1: cached NCM 1.67× (WMT14) and 1.09× (XSum) slower than eager AR, so the CPU hypothesis is confirmed. **NCM compiled too (2026-11-13):** CUDA graphs on the PR #5 cached passes, quality unchanged. Both compiled, interleaved, NCM ÷ AR: WMT14 0.60 / 0.68 / 1.70 and XSum 0.81 / 1.16 / 1.98 at batch 1 / 8 / 32. NCM wins at small batches, AR at batch 32. Dynamo's recompile limit (8) silently fell back to eager in a first run; now guarded. Phase 8 is complete; next is Phase 9. |
| Phase 9, long outputs | **Done (2026-11-13)**, see `docs/phase9.md`. The 1,024-token OWT checkpoint is unreleased, so quality is at 128 tokens (real TinyStories NCM) and speed above 128 uses the OWT architecture with random weights; bf16, everything compiled, interleaved. Quality: genPPL 16.5 vs 10.0 (TinyStories-33M), entropy within 0.011 nats, MAUVE 0.80 vs 0.95 (human ceiling 0.96). Batch 1: NCM 6–10× faster than GPT-2 small and 13–16× than GPT-2 medium at 128–1,024 tokens; the ratio does not improve with length. Batch 8: NCM ÷ GPT-2 small 0.20 → 0.44 → 0.90 at 128 / 256 / 512; 1,024 at batch 8 spills past 6 GB. Success criterion met for speed at batch 1, untested for quality at 1,024. Next: Phase 10. |
| Phase 10, level diagnostic | **Done (2026-11-14)**, see `docs/phase10.md`. Coherence breaks at the finest level: with levels 1–5 real, generating only the 64-code level takes GPT-2 ppl from 15.5 to 111. Greedy there still gives 58 and the generator's top code is right only 36% of the time with perfect context, so independent sampling explains about a third of the damage and capacity/data the rest (hypothesis partly refuted). All levels at T=0.7: 372 → 144 with real-level distinct-2. Next: Phase 10 tokenizer and AR ladder. |
| Phase 10, scale-up | **Done, verdict stop (2026-11-15)**, see `docs/phase10.md`. Same data, BPE and 200M positions: AR ladder 1.9–35.9M reaches MAUVE 0.59–0.92 (human 0.95); NCM 35M gets 0.006 at 61 ms vs 694–1610 ms. Tokenizer reconstruction alone scores 0.86; generating only the last level with oracle levels 1–8 drops it to 0.01 (greedy ppl 61, 4× ar2m's 15.3). Kill rule met: Phase 11 skipped, next Phase 12. |
| Phase 11, attention replacements | **Running (from 2026-11-16), exploratory.** The Phase 10 kill rule skips it; it runs anyway at the author's request, framed as "does the generator need full attention?" rather than as a path to a competitive NCM. Variants `nocross`, `conv`, `linear` plus an attention baseline at 100M positions; speed at 256/1024/2048 on GPU and CPU. Mamba only if conv and linear both fail. |
| Phase 12, SAMSum fine-tuning | **Done (2026-11-16)**, see `docs/phase12.md`. 3 seeds, all 819 test dialogues. ROUGE-L zero-shot / LoRA r8 qv / LoRA r16 all / full: NCM 9.9 / 22.1 / 24.9 / 25.8, BART 15.4 / 36.5 / 39.3 / 39.7. LoRA r16 vs full (paired bootstrap): BART −0.41 [−0.94, +0.12], criterion met; NCM −0.90 [−1.34, −0.46], met on the point estimate only. Fine-tuned NCM still invents a name in 36% of summaries (BART 2.9%, the human rate). |
| Phase 10 extras | **Queued (2026-11-16):** seeds 1–2 for gen25m and ar25m, 7 decoding ablations, blind qwen2.5:3b judge (Ollama, 1,800 items with a real-vs-shuffled check). |
| Fix 6, true fp16 weights | **Done (2026-11-13).** `ncm-kv-half`: quality unchanged (WMT14 +0.01 BLEU, XSum −0.02 ROUGE-L), peak memory at batch 32 3.24 → 2.35 GB (WMT14) and 2.20 → 1.33 GB (XSum), batch-32 throughput +10% / +25%. See `docs/phase4-results.md`. |
| Limits write-up | **Drafted (2026-11-13)** in `docs/limits.md` (prediction verdicts) and the paper's limits paragraphs, abstract and discussion. Needs your edit; the suite predates the cache fix. |
| Paper draft | Phase 4 table, cache paragraph, RAG section, abstract and limitations updated to full-test-set numbers. Still open: the limits-suite "TODO: interpretation" paragraphs and the abstract's limits sentence. |
| Citation | **Done.** `paper/refs.bib` uses the authors' BibTeX (see below). |

**Contact from the authors (2026-11-12).** Hermann Kumbong (first author) wrote to say
thanks for the benchmark and to point out that the reference lacked its author list; the
full BibTeX is at github.com/HazyResearch/ncm#citation. That entry is now in
`paper/refs.bib` and the README credits "Kumbong et al.". This changes three things in the
plan:

1. **Upstream KV-cache report.** PR #5 already has the fix. Post the RTX 3050 laptop
   results (fp32 identity, fp16 BLEU −0.13, batch-1 vs batch-32 speedup) as a comment on it,
   and point Hermann to it in your reply. A new issue would only duplicate it.
2. **Pretraining confound (fix 4, step 1).** Ask directly whether the matched-data AR
   baselines (ROUGE-L 24.4, BLEU 25.2) can be released. That one question could replace a
   90-hour retrain.
3. **Tone of the write-up.** The authors will probably read it. Keep the negative results,
   but word them as measured on this hardware and at this output length; the paper's
   headline numbers are against diffusion LMs at 1,024 tokens, which Phase 9 is the first
   to test here. Offer them a draft before publishing; they may catch setup errors.

## Assumptions and things not checked

- **Paper facts.** The plan was first written from the repo docs only. The official code is
  now in `third_party/ncm`; check the facts marked *(paper, via repo)* against arXiv
  2610.14162 before the write-up.
- **Time:** 10 hands-on hours per week for 12 weeks, plus unattended overnight GPU runs
  (about 40 GPU-hours per week). With 5 h/week, drop Phase 11.
- **Cloud budget:** none. One optional item costs about $15–30.
- **Machine:** 16 GB RAM and 40 GB free disk; laptop plugged in, fixed power mode.
- **Training throughput:** about 15k tokens/s for a 25M model in bf16. This is an estimate;
  run a 10-minute probe first and rescale every token budget to it.
- **Limits suite** has now run (see Status); the `python: command not found` log was from
  the first attempt.

## Measurement protocol (applies to every phase)

- **Latency:** 20 warm-up calls per input shape, `torch.cuda.synchronize()` around the
  clock, 200 timed calls, models interleaved (A, B, A, B) so thermal throttling hits both.
  Report median and p95 with bootstrap 95% intervals, and log GPU temperature.
- **Caching parity:** each model runs with its best working cache. If one side has none, say
  so in the table header.
- **Quality:** paired bootstrap over examples, 10,000 resamples (sacreBLEU has this built in
  as `--paired-bs`). A difference counts only if the 95% interval excludes zero.
- **Training comparisons:** 3 seeds for headline pairs; claim a difference only if it exceeds
  twice the seed standard deviation.
- **Binary outcomes (RAG):** Wilson intervals and McNemar's test on paired answers.

## Prioritized fixes to existing results

| # | Fix | Why it comes here | Cost |
|---|---|---|---|
| 1 | Run the limits suite (fix the `python` call in WSL) | Eight experiments are designed but empty | 2 GPU-h |
| 2 | Evaluate on full test sets: XSum 11,334, WMT14 3,003, all 1,190 XQuAD questions | Removes the small-sample weakness outright; n=40 gives about ±15 points | ~1.5 GPU-h |
| 3 | KV-cache diagnosis and prefix caching (Phase 8) | Explains both the BLEU 3.8 bug and the batch-32 slowdown | 8 h hands-on |
| 4 | Pretraining confound | See below | – |
| 5 | Long outputs (Phase 9) | The only setting where the paper claims a large win | 3 GPU-h |
| 6 | True fp16 weights (`.half()`), fall back to bf16 if QK-norm overflows | Current memory numbers are inflated | 1 h |
| 7 | Add exact-match and token-F1 RAG scoring beside string containment | Lenient scoring | 1 h |
| 8 | Re-run multilingual RAG with the fixed German splitter | Open TODO in the paper | 0.5 GPU-h |
| 9 | Diffusion baseline | Skip. It only re-tests the 31× claim, which is not the question here | – |

**Pretraining confound (fix 4), recommendation:** do not retrain at official scale locally.
A from-scratch WMT14 AR model needs roughly 2.5B training tokens, about 90 hours on this GPU
(or about $15–30 on a rented RTX 4090). Instead:

1. Check whether the paper's matched AR baselines are released. They score ROUGE-L 24.4 and
   BLEU 25.2 *(paper, via repo)*.
2. For speed, build a randomly initialized AR encoder-decoder with the same width, depth and
   parameter count as the NCM generator and force its output length. Latency does not depend
   on the weights.
3. For quality under full control, rely on Phase 10.

## Phase 8: repair the official-checkpoint benchmark (weeks 1–3)

**Question:** how much of the batch-1 tie and batch-32 loss is the method, and how much is
missing caching and overhead? It comes first because every later speed claim reuses this
harness.

**Experiments**

- **Cache bisect.** One WMT14 sentence, fp32, batch 1, no padding. Compare per-level logits
  with `use_kv_cache` on and off and find the first level and layer that diverge. Then repeat
  with a padded batch of 8.
- **Prefix cache.** If the prefix does not attend to the levels, its keys and values can be
  computed once and reused across all 9–11 passes. *Hypothesis:* this removes most of the
  7.6 s of 8.0 s per batch on XSum at batch 32 *(repo)*.
- **Optimized AR.** Add `torch.compile(mode="reduce-overhead")` and a static cache to BART
  and opus-mt, and to NCM too.
- **CPU run.** Same benchmark at batch 1 on CPU with 4 threads. *Hypothesis:* NCM loses on
  CPU, because it processes about twice as many positions in total and the GPU's idle
  parallel capacity no longer hides that.
- **Vectorize** the penalty loop (411 ms at batch 32) and the codes-to-logits stage (730 ms
  on WMT14).

**Held constant:** full test sets, greedy decoding, fp16 weights, same prefix lengths.

**Success:** cached NCM output matches uncached output to within 0.1 BLEU, and the batch-32
gap is re-measured.

**Kill:** if the checkpoint was trained with attention that makes caching invalid, stop after
8 hours, report that, and time NCM uncached with a clear label.

**Misleading-result risk:** a cache that is "fast and nearly right". Always report quality
next to every speed number.

**Deliverables:** `bench/` cache-parity tests, updated Phase 4 tables, the upstream issue.

### Upstream KV-cache report

1. Reproduce on a clean clone at a pinned commit using only the official `evaluate.py`,
   toggling only the flag.
2. Search existing issues first.
3. Write a script under 40 lines: load the WMT14 checkpoint, translate 8 fixed sentences both
   ways, print the outputs and BLEU on 500 sentences (20.0 vs 3.8).
4. Include the bisect result (first diverging level, padded vs unpadded, fp32 vs fp16),
   commit hash, PyTorch and CUDA versions.
5. Offer a pull request if the cause is found.

## Phase 9: long outputs (week 4)

**Question:** does NCM win at batch 1 when outputs are long? The paper's timings use
1024-token outputs on unconditional checkpoints *(paper, via repo)*, which were skipped.

- **Setup:** download the unconditional OpenWebText checkpoint. Compare against GPT-2 small
  (124M) with a KV cache, generating 128, 256, 512 and 1024 tokens at batch 1 and 8, fp16.
- **Quality:** 200 samples each; perplexity under GPT-2 large, plus unigram entropy and
  MAUVE. Perplexity alone rewards repetitive text.
- **Success:** NCM at least 3× faster at 1024 tokens with entropy within 0.2 nats of GPT-2's.
- **Kill:** if the checkpoint does not fit in 6 GB at batch 1 in fp16, run speed only on a
  rented GPU for about $2, or drop the phase.
- **Risk:** comparing nucleus-sampled GPT-2 with greedy NCM. Fix one sampling setting per
  model and report it.
- **Deliverable:** latency against output length plot with the measured break-even length.

## Phase 10: controlled comparison and scale-up (weeks 5–8, goals 1 and 3)

**Question:** at equal data, tokenizer and budget, does NCM sit above the AR quality-latency
frontier, and does a 25M NCM become fluent?

**Diagnostics first (no training, 1 evening).** The toy tokenizer reconstructs 98.7%
*(repo)*, so the incoherence is in the generator. Feed real codes for levels 1 to k and
generate the rest, for each k. This shows which level breaks coherence. *Hypothesis:* the
cause is independent sampling within a level, not capacity, so parameters alone will help
only partly.

**Setup**

- **Data:** full TinyStories (about 2.1M stories, roughly 450M tokens), 256 tokens per
  example, the existing 4096-token BPE. Phase 2 used only about 19M tokens, which is itself a
  likely cause of poor fluency.
- **NCM:** levels 1, 2, 4, …, 256 (9 levels). Tokenizer about 10M trained for 100M tokens;
  generators of 12M and 25M trained for 200M tokens.
- **AR ladder:** 1.5M, 3M, 7M, 25M and 35M, each for 200M tokens. The 35M model matches
  NCM's total footprint (generator plus tokenizer).
- **Constant:** optimizer, schedule, bf16, batch 64, seed set, sampling temperature tuned per
  model on a validation set.
- **Decoding ablations on the 25M NCM:** two refinement passes on the last two levels,
  per-level temperature, classifier-free guidance.
- **Seeds:** 3 for the 25M NCM and the 25M AR; 1 for the rest.

**Metrics:** GPT-2 perplexity, MAUVE, distinct-2, and a blind 1–5 coherence score from
qwen2.5:3b on 200 samples. Check first that the judge separates real stories from
sentence-shuffled ones. Latency at batch 1 on GPU and CPU.

**Success:** some NCM point is faster than every AR model of equal or better quality.

**Kill:** if the 25M NCM with the best decoding is still above 3× the perplexity of the AR
model that matches its latency, stop scaling. Report a negative result and move to Phase 12,
where conditional tasks are known to work.

**Risk:** an unoptimized AR loop. The Phase 2 AR model took about 5 ms per token for a 7M
model, which is mostly Python overhead, so the 9.3× figure is close to the pass-count ratio
(64 / 6) and may shrink a lot under `torch.compile`.

**Realism check:** a 25M model trained on TinyStories will not be a generally useful language
model, whatever the architecture. The achievable version of goal 3 is "fluent within a narrow
domain".

**Deliverables:** Pareto plot, scale curve, level-localization figure.

## Phase 11: attention replacements (weeks 9–10, goal 2)

**Question:** can the generator drop full attention without losing quality? It comes after
Phase 10 because it needs that tokenizer and baseline.

- **Setup:** frozen Phase 10 tokenizer, 25M generators, 100M tokens each, in this order:
  1. No cross-level attention (a mask change only). Each level's input already contains the
     canvas so far, so cross-level attention may be redundant.
  2. Depthwise convolution blocks on levels of 64 positions or more, attention kept on coarse
     levels.
  3. Linear attention.
  4. Bidirectional Mamba, only if 2 and 3 fail and time remains. It adds a compiled
     dependency.
- **Recommendation:** variant 2 is the main bet. It has no new dependency and helps CPU most.
- **Speed:** measure at 256 positions, and at 1024 and 2048 with random weights, on GPU and
  CPU.
- **Success:** quality within seed noise of the baseline and at least 1.3× faster or leaner
  at 1024 positions.
- **Kill:** if no variant is 1.2× faster at any length up to 1024, write it up as "attention
  is not the bottleneck at this scale".
- **Risk:** half-budget training hides slow-converging variants. Re-run the top two at 200M
  tokens with 2 seeds.

## Phase 12: fine-tuning for a narrow task (week 11, goal 4)

**Question:** what is the cheapest way to adapt a released checkpoint? It only depends on
Phase 8.

- **Task:** SAMSum dialogue summarization (14.7k train, 819 test). It fits the XSum
  checkpoint's 512-in, 48-out canvas.
- **Gate:** the frozen VQ tokenizer must reconstruct SAMSum summaries at 95% token accuracy
  or better. If not, add a decoder fine-tuning arm.
- **Arms, 3 seeds each, 3 epochs:** zero-shot; LoRA rank 8 on query and value; LoRA rank 16
  on all attention and MLP layers; full generator fine-tune. Run the same arms on
  `bart-base-xsum`, which starts from the same XSum-tuned point.
- **Metrics:** ROUGE-L on all 819 test dialogues, the faithfulness proxies, batch-1 latency.
- **Success:** a LoRA arm within 1 ROUGE-L of full fine-tuning.
- **Kill:** if the official repo has no usable training loss and rebuilding it takes more
  than 6 hours, fine-tune the Phase 10 NCM on a TinyStories subset instead.
- **Risk:** BART is still pretrained, so frame this as a practitioner's choice, not a method
  comparison. Check the checkpoint licence before sharing weights.

## Resource estimates

| Experiment | Peak VRAM | GPU time | Hands-on |
|---|---|---|---|
| Limits suite and full test sets | 3.8 GB | 4 h | 4 h |
| Phase 8 cache, fp16, compile, CPU | 4 GB | 3 h | 16 h |
| Phase 9 long outputs | 4.8 GB at batch 1, 1,024 tokens, three models loaded; batch 8 at 1,024 spills (measured) | 3 h | 6 h |
| Phase 10 tokenizer and diagnostics | 3 GB | 4 h | 6 h |
| Phase 10 AR ladder (5 models) | 3 GB | 12 h | 3 h |
| Phase 10 NCM 12M and 25M | 4 GB | 13 h | 4 h |
| Phase 10 extra seeds and ablations | 4 GB | 26 h | 4 h |
| Phase 11 four variants and re-runs | 4 GB | 32 h | 10 h |
| Phase 12 eight arms × 3 seeds | 4–5 GB | 10 h | 8 h |

## Schedule

| Week | Work | Output |
|---|---|---|
| 1 | Run limits suite, full test sets, statistics harness | Filled Phase 7 tables with intervals |
| 2 | Cache bisect, true fp16, upstream issue | Minimal reproduction posted |
| 3 | Prefix cache, compiled AR, CPU run | Corrected Phase 4 tables |
| 4 | Long-output benchmark | Break-even length plot |
| 5 | Throughput probe, diagnostics, tokenizer | Level-localization figure |
| 6 | AR ladder | AR frontier |
| 7 | NCM 12M and 25M, decoding ablations | First Pareto plot |
| 8 | Extra seeds, go or stop decision | Phase 10 write-up |
| 9 | Variants 1 and 2 | Ablation table |
| 10 | Variant 3, re-runs of top two | Phase 11 write-up |
| 11 | SAMSum fine-tuning | Adaptation table |
| 12 | Report, blog post, reproduce script | Release |

## Write-up and strongest honest claims

**Format:** publish a technical report in the repo and a blog post first. Go to arXiv only if
Phase 10 gives a clean three-seed result; an independent author needs an endorser for cs.CL.
Rename the draft: it should not say "fair benchmark" until Phase 8 lands. Keep the
pre-registered predictions and the AI-assistance statement. Cite the paper with the authors'
BibTeX (done) and send them the draft before it goes public.

| Outcome | Strongest honest claim |
|---|---|
| NCM beats the AR frontier in Phase 10 and wins at long outputs | "At 25M parameters and matched training, NCM gives X× lower batch-1 latency at equal quality on a 6 GB GPU, for outputs of N tokens or more." |
| Wins on GPU, loses on CPU | "NCM's advantage is unused GPU parallelism at batch 1; it does not carry over to CPU-only devices." |
| Faster but never at matched quality | "A smaller AR model is both faster and more fluent; NCM's speed needs a quality cost at this scale." |
| Phase 8 cache fix closes the batch-32 gap | "The released checkpoints' throughput deficit is an implementation gap, not a property of the method." |
| Attention replacement works | "Cross-level attention (or fine-level attention) is removable at no measurable cost at this scale." Limit it to 25M parameters and 256 tokens. |
| LoRA matches full fine-tuning | "Released NCM checkpoints adapt to a new summarization domain with under 1% trainable parameters." |

In every case the claim is limited to one GPU, toy scale for the controlled part, and task
checkpoints trained by others.
