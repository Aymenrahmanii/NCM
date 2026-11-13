# Phase 4: testing the speed claim fairly

## Predictions (written 2026-10-26, before running the benchmark)

1. **NCM beats the AR baseline on latency at batch size 1** on both tasks, but by much less
   than 31×: the 31× figure is against a diffusion LM (AURORA-LM, 128+ steps), not against an
   AR model with a KV cache. Guess: 2–5×.
2. **NCM's advantage shrinks as batch size grows.** At batch 1 the AR model wastes the GPU on
   tiny per-token passes; at batch 32 each AR pass has more parallel work, so the gap closes.
3. **NCM's advantage is larger on XSum than on WMT14** if the AR summaries are longer than the
   AR translations (more sequential steps to avoid). It may also be smaller because XSum
   NCM reads a 512-token prefix in every one of its 8 passes.
4. **Quality:** NCM is close to BART-base on XSum ROUGE-L (paper: NCM 28.0 vs its own AR 24.4),
   but loses on WMT14 BLEU to opus-mt-de-en, which was trained on far more parallel data than
   WMT14 alone (paper: NCM 22.6 vs AR 25.2 even with matched data).
5. **Memory:** NCM uses more GPU memory than the AR baselines, because it carries a separate
   VQ-VAE tokenizer (60M for XSum, 144M for WMT14) on top of the generator.
6. **fp16 vs fp32:** small speedup, near-identical quality.

## Setup

- GPU: RTX 3050 6 GB laptop (WSL2), PyTorch 2.14.1+cu130.
- Same 500 test examples per task, same precision, seed 0, greedy decoding for every model.
- Timing covers generation only (`run`), after warm-up, with `torch.cuda.synchronize()`.
- XSum: NCM (88M generator + 60M VQ-VAE) vs `morenolq/bart-base-xsum` (140M). The AR model
  reads the same 512-token article prefix as text.
- WMT14 De→En: NCM (120M generator + 144M VQ-VAE) vs `Helsinki-NLP/opus-mt-de-en` (74M).

## Results

Run: `python -m ncm_lab.bench.run --task xsum|wmt14 --n 500 --precision fp16`, then
`python -m ncm_lab.bench.plot`. Raw numbers: `results/phase4_*_fp16.csv`; every output next to
its reference: `results/phase4_outputs_*.jsonl`; charts: `results/phase4_*_fp16.png`.

### Full test sets with 95% intervals (rerun 2026-11-11)

Same setup on every test example: XSum 11,334 articles, WMT14 3,003 sentences. Quality
intervals come from a paired bootstrap (10,000 resamples, `python -m ncm_lab.bench.significance`,
full table in `results/significance.md`). Raw numbers are in `results/phase4_*_fp16.csv`.

| | XSum NCM | XSum BART-base | WMT14 NCM | WMT14 opus-mt |
|---|---|---|---|---|
| quality | ROUGE-L 27.94 [27.68, 28.18] | ROUGE-L 33.26 [32.97, 33.55] | BLEU 21.73 [21.19, 22.24] | BLEU 33.05 [32.36, 33.72] |
| NCM − AR | **−5.33 [−5.56, −5.09]** | | **−11.32 [−11.85, −10.80]** | |
| ROUGE-1 / 2 | 34.92 / 12.72 | 41.31 / 18.01 | – | – |
| latency b=1, median / p95 (ms) | 243 / 277 | 236 / 320 | 244 / 296 | 227 / 387 |
| throughput b=1 / 8 / 32 (ex/s) | 4.1 / 5.4 / 5.2 | 4.1 / 20.5 / **50.6** | 4.1 / 17.8 / 19.1 | 4.3 / 17.6 / **49.4** |
| peak GPU memory b=32 | 2.9 GB | 0.65 GB | 3.8 GB | 0.24 GB |

What changes against the n=500 tables below: every quality gap is significant, and the
intervals are about ±0.3 ROUGE-L and ±0.5 BLEU. At batch 1 these single runs put NCM level
with AR on median latency for both tasks; the earlier 1.2× WMT14 lead does not hold on the
full set. NCM's p95 is still tighter. At batch 32, AR is 2.6× (WMT14) and 9.8× (XSum) faster.
These latencies have no interval; the interleaved run below replaces them and finds that
opus-mt is in fact 1.26× faster than NCM at batch 1 on WMT14.

**KV cache (`results/cache_parity/`, `python -m ncm_lab.bench.cache_parity`).** The BLEU 3.8
with `use_kv_cache=True` (see below) is a padding bug in the cached path, not a property of
the method. In fp32 the cached and uncached outputs match on 500 of 500 sentences. With the
pad fix in fp16, cached BLEU is 19.94 against 20.07 uncached (difference −0.13, 95% CI
[−0.25, −0.03]); the first divergence is at level 6, block 0, about 1% relative, which fits
fp16 rounding. Speed: 0.86× at batch 1, where launches dominate, and 1.87× at batch 32.

**Upstream fix, PR #5 (`results/upstream_pr5/`).** Upstream PR #5 fixes the same mask bug,
reuses the prefix K/V from the level-0 pass, and enables the cache for XSum. Its
`scripts/bench_kv.py` at commit `60c947b`, run on this GPU in fp32 with TF32 off. Each run
is a single timed pass, so there are no intervals.

| run | eager | KV cache | speedup | identical | metric (both) | peak memory eager / KV |
|---|---|---|---|---|---|---|
| WMT14, 1,024 at batch 16 | 92.5 s | 37.6 s | **2.46×** | 100% | BLEU 20.59 | 2.14 / 2.73 GB |
| WMT14, 128 at batch 1 | 30.9 s | 31.3 s | 0.99× | 100% | BLEU 17.00 | 1.21 / 1.25 GB |
| XSum, 256 at batch 16 | 104.0 s | 23.2 s | **4.48×** | 100% | ROUGE-L 27.85 | 2.02 / **4.92 GB** |

The cache removes the repeated prefix work that made XSum NCM's worst case at batch 16.
At batch 1 it gains nothing: launch overhead dominates, not prefix compute. So the batch-1
tie with the AR models stands, and the batch-32 deficit needs re-measuring with the cache on.
XSum memory more than doubles with the cache in this fp32 run, which did not include PR #4.

**Phase 4 rerun with the cache on (`ncm-kv`, 2026-11-12).** Same harness, full test sets,
fp16. Upstream code is PR #5 with PR #4 (the `generate()` memory leak fix) on top, via
`NCM_ROOT=third_party/ncm-pr5-pr4`. Rows are in `results/phase4_*_fp16.csv` next to the
uncached ones, and the log is `results/upstream_pr5/phase4_kv_run.log`.

| | NCM uncached | **NCM + cache** | AR |
|---|---|---|---|
| WMT14 BLEU | 21.73 | 21.70 | 33.05 |
| WMT14 throughput b=1 / 8 / 32 (ex/s) | 4.1 / 17.8 / 19.1 | 4.1 / **29.7** / **49.5** | 4.3 / 17.6 / 49.4 |
| WMT14 peak memory b=32 | 3.8 GB | 3.2 GB | 0.24 GB |
| XSum ROUGE-L | 27.94 | 27.94 | 33.26 |
| XSum throughput b=1 / 8 / 32 (ex/s) | 4.1 / 5.4 / 5.2 | 4.0 / **21.1** / **23.1** | 4.1 / 20.5 / 50.6 |
| XSum peak memory b=32 | 2.9 GB | 2.2 GB | 0.65 GB |

- **Quality does not move.** Cached minus uncached, paired bootstrap: WMT14 BLEU −0.03
  [−0.08, +0.03], with 2,834 of 3,003 outputs identical. XSum ROUGE-L +0.00 [−0.00, +0.01],
  with 11,046 of 11,334 identical. The few differing outputs come from fp16 rounding; in
  fp32 the outputs are identical.
- **Most of the batch-32 deficit was missing caching, not the method.** WMT14 goes from
  2.6× slower than opus-mt to level with it, and is 1.7× faster at batch 8. XSum goes from
  9.8× slower than BART to 2.2× slower, and is level with it at batch 8.
- **Batch 1 does not change** with the cache (about 4 ex/s for every model). The interleaved
  run below refines the batch-1 comparison with AR.
- **Memory drops below the uncached run** once PR #4 is in. So the 4.9 GB fp32 figure above
  came mostly from the leak, not from the cache.
- **Caveat:** these are single runs, and the NCM-cache and AR rows were measured on
  different days, not interleaved. GPU at 79–83 °C with the SM clock steady at 1,492 MHz.
  Treat WMT14 batch 32 as a tie, not a win.
- **The quality gap is unchanged.** NCM is still 11.3 BLEU and 5.3 ROUGE-L behind the
  pretrained AR models, and caching does not touch that.

**True fp16 weights (`ncm-kv-half`, 2026-11-13, plan fix 6).** All rows above keep NCM's
weights in fp32 and only autocast the compute, while the AR baselines load fp16 weights
(`torch_dtype`). `NCMModel(cast_weights=True)` stores generator and VQ-VAE weights in fp16 as
well; autocast stays on, so norms and softmax still run in fp32. QK-norm did not overflow, so
the bf16 fallback was not needed. Same harness, full test sets, KV cache on. Log:
`results/phase4_half_run.log`.

| | NCM + cache, fp32 weights | **NCM + cache, fp16 weights** | AR |
|---|---|---|---|
| WMT14 BLEU | 21.70 | 21.71 | 33.05 |
| WMT14 throughput b=1 / 8 / 32 (ex/s) | 4.1 / 29.7 / 49.5 | 3.9 / 29.6 / **54.4** | 4.3 / 17.6 / 49.4 |
| WMT14 peak memory b=1 / b=32 | 1.51 / 3.24 GB | **0.61 / 2.35 GB** | 0.15 / 0.24 GB |
| XSum ROUGE-L | 27.94 | 27.92 | 33.26 |
| XSum throughput b=1 / 8 / 32 (ex/s) | 4.3 / 20.6 / 23.3 | 3.9 / 22.9 / **29.0** | 4.1 / 20.5 / 50.6 |
| XSum peak memory b=1 / b=32 | 0.94 / 2.20 GB | **0.39 / 1.33 GB** | 0.29 / 0.65 GB |

- **Quality does not move.** fp16 minus fp32 weights, paired bootstrap: WMT14 BLEU +0.01
  [−0.07, +0.09] (2,769 of 3,003 outputs identical); XSum ROUGE-L −0.02 [−0.05, −0.00]
  (10,091 of 11,334 identical). The XSum interval just excludes zero, but 0.02 points is
  far below anything that matters.
- **Memory was inflated by the fp32 weights.** Weights alone drop from 1.08 to 0.56 GB
  (WMT14) and 0.69 to 0.36 GB (XSum). NCM's memory is now 1.3–4× the AR models' at batch 1
  and 2–10× at batch 32, instead of 3–16×.
- **Speed: faster at batch 32, slightly slower at batch 1.** Interleaved head-to-head
  (`--models ncm-kv ncm-kv-half --batch-sizes 1 32 --tag weights`), fp32-weight ÷ fp16-weight
  median latency: XSum batch 32 **1.15** [1.13, 1.17], batch 1 0.97 [0.95, 1.00]; WMT14
  batch 1 **0.96** [0.93, 0.99], batch 32 1.18 [1.14, 1.22]. So fp16 weights cost 3–4% at
  batch 1 and save about 15% at batch 32. Treat the WMT14 batch-32 figure as suspect: both
  models' absolute times in that run (1,158 and 983 ms) are far above every other WMT14
  run (about 620–650 ms) at similar temperature and clock. The likely but unverified cause
  is memory pressure from holding both copies of the weights near the 6 GB limit.
- **Penalty loop.** The XSum rows in this table (both NCM columns) use a rewritten
  frequency/adjacency penalty loop (`ncm_lab.models.ncm.pick_tokens`): the penalty terms are
  updated at the picked tokens instead of rebuilt over the whole vocabulary every step, and
  logits are cast to float64 one step at a time. It picks exactly the tokens upstream's loop
  picks (unit tests, and all 11,334 XSum outputs byte-identical to the previous run). On
  real batch-32 logits it takes 63 instead of 120 ms and 62 instead of 640 MB of scratch
  memory (batch 1: 17 vs 23 ms). That is about 4% of a cached batch-32 XSum batch. The old
  loop alone takes 120 ms here, not the 411 ms in the profile below; that profile is from an
  earlier run (different code and GPU state) and should be redone before it is quoted. The
  fp16 throughput gain above comes from the weights.

### Latency under the full protocol, interleaved (2026-11-12)

`NCM_ROOT=third_party/ncm-pr5-pr4 python -m ncm_lab.bench.interleaved --task wmt14|xsum`.
Per batch size: 20 warm-up calls per model, then 200 rounds. Each round runs NCM uncached,
NCM + cache and AR once on the same input batch, in an order that rotates every round.
`torch.cuda.synchronize()` brackets each call. Inputs come from the full test sets and are
tokenized before the clock starts. Intervals are bootstrap 95% (10,000 resamples). The
ratio interval is a paired bootstrap over rounds, so drift within a round cancels. fp16
autocast. Files: `results/phase8_interleaved_{wmt14,xsum}{.csv,_ratios.csv,_calls.jsonl}`.

Median latency in ms [95% CI] (p95):

| | NCM | NCM + cache | AR |
|---|---|---|---|
| WMT14 b=1 | 232 [226, 236] (283) | 234 [229, 240] (290) | **185** [170, 202] (387) |
| WMT14 b=8 | 390 [386, 393] | **261** [256, 265] | 460 [432, 474] |
| WMT14 b=32 | 1,510 [1,504, 1,519] | 650 [648, 656] | 629 [601, 650] |
| XSum b=1 | 241 [238, 246] (302) | 229 [224, 235] (291) | 244 [229, 256] (344) |
| XSum b=8 | 1,561 [1,549, 1,580] | 398 [393, 402] | 409 [402, 420] |
| XSum b=32 | 6,241 [6,215, 6,286] | 1,418 [1,411, 1,429] | **696** [684, 709] |

NCM ÷ AR median latency, paired [95% CI]. Above 1 means AR is faster.

| | WMT14 NCM | WMT14 + cache | XSum NCM | XSum + cache |
|---|---|---|---|---|
| batch 1 | 1.26 [1.14, 1.37] | 1.27 [1.15, 1.38] | 0.99 [0.94, 1.05] | 0.94 [0.89, 1.00] |
| batch 8 | 0.85 [0.82, 0.90] | **0.57** [0.55, 0.61] | 3.82 [3.72, 3.89] | 0.97 [0.95, 0.99] |
| batch 32 | 2.40 [2.32, 2.51] | 1.03 [1.00, 1.08] | 8.97 [8.79, 9.14] | 2.04 [2.00, 2.08] |

- **Batch 1 is not a tie on WMT14.** opus-mt's median is 1.26× lower, and the interval
  excludes 1. On XSum batch 1 is a tie: 0.99 uncached, 0.94 cached, which only just
  excludes 1. The earlier "level at batch 1" for WMT14 came from single non-interleaved runs.
- **NCM's tail is tighter at batch 1** on both tasks: p95 283 vs 387 ms (WMT14), 302 vs
  344 ms (XSum). AR's p95 is about 2× its median on WMT14, because opus-mt's output length
  varies with the input.
- **With the cache, NCM wins only at batch 8 on WMT14** (1.8× faster). At batch 32 WMT14 is
  a tie with AR 3% ahead at the edge of significance; XSum is level at batch 8 and 2.0×
  behind at batch 32. That matches the single-run throughput table above (2.2×).
- **The cache does nothing at batch 1** (WMT14 0.99 [0.96, 1.02]) and gives 1.5–2.3×
  (WMT14) and 3.9–4.4× (XSum) at batches 8 and 32.
- **Conditions.** The GPU ran at 66–88 °C with the SM clock between 1.2 and 1.9 GHz. The
  rotating order spreads that drift over all three models; compare ratios, not absolute
  times across tables or runs. The first XSum attempt stopped without an error at the start
  of batch 32 (cause unknown; WSL had restarted by the time it was noticed), so the XSum
  numbers are from a clean rerun.

### Compiled AR baselines (2026-11-13, Phase 8)

The AR rows so far use Hugging Face `generate` in eager mode, which at batch 1 is mostly
Python and kernel-launch overhead. `ar-compiled` is the same model and weights with
`cache_implementation="static"`, so `generate` compiles the decoding step
(`torch.compile`, CUDA graphs). Source lengths are padded to multiples of 64 to bound the
number of compiled shapes, and the harness warms compiled models up on every distinct input
batch, so no compilation lands inside a timed call. **NCM is not compiled** in any row:
compiling its level loop is still open, so this compares the best AR setup against
NCM eager + cache + fp16 weights.

Quality is unchanged: BLEU 33.06 vs 33.05 (2,986 of 3,003 outputs identical), ROUGE-L 33.25
vs 33.26 (11,136 of 11,334 identical). Single-run Phase 4 harness, full test sets:

| | NCM + cache, fp16 weights | AR eager | **AR compiled** |
|---|---|---|---|
| WMT14 throughput b=1 / 8 / 32 (ex/s) | 3.9 / 29.6 / 54.4 | 4.3 / 17.6 / 49.4 | **9.9 / 47.6 / 126.2** |
| XSum throughput b=1 / 8 / 32 (ex/s) | 3.9 / 22.9 / 29.0 | 4.1 / 20.5 / 50.6 | **8.3 / 35.3 / 62.7** |

Interleaved, full protocol (`--models ncm-kv-half ar ar-compiled --tag compiled`; files
`results/phase8_interleaved_{wmt14,xsum}_compiled*`). Median ms [95% CI]:

| | NCM + cache, fp16 w. | AR eager | AR compiled |
|---|---|---|---|
| WMT14 b=1 | 288 [282, 293] | 219 [204, 247] | **100** [91, 107] |
| WMT14 b=8 | 304 [298, 310] | 519 [491, 549] | **174** [166, 177] |
| WMT14 b=32 | 622 [613, 629] | 690 [680, 709] | **277** [263, 284] |
| XSum b=1 | 260 [256, 265] | 276 [260, 282] | **119** [116, 122] |
| XSum b=8 | 340 [335, 346] | 439 [426, 446] | **234** [230, 239] |
| XSum b=32 | 1,190 [1,167, 1,198] | 711 [697, 718] | **543** [534, 552] |

NCM ÷ AR-compiled median latency, paired [95% CI]: WMT14 2.88 [2.69, 3.15] at batch 1,
1.75 [1.70, 1.84] at 8, 2.25 [2.19, 2.36] at 32; XSum 2.18 [2.11, 2.25], 1.46 [1.42, 1.49],
2.19 [2.15, 2.22].

- **Compiling AR halves its latency** (2.2–2.3× at batch 1, 1.3–3.0× at larger batches).
  The batch-1 "tie" with NCM was a tie with an overhead-bound AR loop, as the plan's
  misleading-result risk warned.
- **Against compiled AR, NCM is slower at every batch size on both tasks**, by 1.5–2.9×.
  Its only remaining wins are against eager AR (batch 8, both tasks, and WMT14 batch 32
  in this run).
- **Superseded by the next section.** NCM's 9–11 level passes are launch-bound too, and
  compiling them reverses the batch-1 result.
- **Conditions.** GPU 72–86 °C, SM clock 1.36–1.94 GHz across the two runs.

### Both sides compiled (2026-11-13, Phase 8)

`ncm-kv-half-compiled` compiles the two cached passes from upstream PR #5 (`_kv_prefix`,
`_kv_level`) with `torch.compile(mode="reduce-overhead")`, i.e. CUDA graphs; see
`ncm_lab.models.ncm._compile_cached_passes`. Outputs leaving the graphs are cloned, because
the prefix K/V must outlive the level passes that follow. The tokenizer decoder and the
penalty loop stay eager.

**Pitfall, now guarded.** `_kv_level` needs one graph per (level, batch shape): 10 levels
× 4 shapes on WMT14. Dynamo's default `recompile_limit` is 8, and beyond it torch *silently*
runs the remaining levels eagerly. A first run of this comparison hit that and was discarded
(WMT14 batch 1: 91 ms partly eager vs 57 ms fully compiled in a diagnostic). Both compiled
models now raise the limit to 128 and set `fail_on_recompile_limit_hit`, so a fallback
aborts the run instead of timing eager code.

Quality is unchanged. Compiled minus eager NCM, paired bootstrap: WMT14 BLEU −0.00
[−0.07, +0.06] (2,787 of 3,003 identical), XSum ROUGE-L +0.00 [−0.01, +0.02] (10,618 of
11,334 identical). Interleaved, full protocol (`--models ncm-kv-half-compiled ar-compiled
--tag bothcompiled`), median ms [95% CI] (p95):

| | NCM compiled | AR compiled | NCM ÷ AR [95% CI] |
|---|---|---|---|
| WMT14 b=1 | **62** [60, 64] (85) | 104 [98, 109] (184) | **0.60** [0.56, 0.64] |
| WMT14 b=8 | **119** [118, 120] (134) | 175 [164, 180] (253) | **0.68** [0.66, 0.73] |
| WMT14 b=32 | 496 [490, 498] (559) | **292** [282, 300] (381) | 1.70 [1.65, 1.76] |
| XSum b=1 | **92** [89, 93] (123) | 113 [110, 120] (157) | **0.81** [0.76, 0.84] |
| XSum b=8 | 260 [259, 261] (287) | **223** [217, 228] (274) | 1.16 [1.14, 1.20] |
| XSum b=32 | 1,017 [1,011, 1,028] (1,170) | **513** [503, 519] (647) | 1.98 [1.96, 2.02] |

- **With both sides compiled, NCM wins at small batches.** It is 1.67× faster than
  opus-mt at batch 1 and 1.47× at batch 8, and 1.23× faster than BART at batch 1. Its p95
  is also lower everywhere at batch 1 and 8 on WMT14 (85 vs 184 ms at batch 1).
- **AR still wins at batch 32** (1.7× WMT14, 2.0× XSum) and on XSum from batch 8. Once the
  GPU is busy, NCM's roughly twice as many processed positions cost real compute.
- **Compiling helps NCM more than AR at batch 1:** 4.6× for NCM (288 → 62 ms on WMT14,
  eager vs compiled, different runs) against 2.1–2.3× for AR, because each of NCM's level
  passes is a full stack launch over few tokens.
- **The quality gap is untouched** (11.3 BLEU, 5.3 ROUGE-L), so this is speed at lower
  quality against pretrained baselines; Phase 10 is what tests equal quality.
- **Conditions.** Total GPU memory peaked at 5.66 of 6.14 GB (about 1.9 GB is the Windows
  desktop), so there was no spill; 67–88 °C. Log: `results/phase8_ncm_compile_gpu.csv`.

### CPU, batch 1, 4 threads (2026-11-13, Phase 8)

`--device cpu --threads 4 --precision fp32 --batch-sizes 1 --calls 100 --warmup 5 --tag cpu4`
(WSL, 16 logical cores available, fp32 because fp16 is slow on CPU). Files:
`results/phase8_interleaved_{wmt14,xsum}_cpu4*`. Median ms [95% CI]:

| | NCM | NCM + cache | AR eager |
|---|---|---|---|
| WMT14 | 1,842 [1,828, 1,856] | 956 [947, 968] | **574** [514, 635] |
| XSum | 5,236 [5,195, 5,254] | 1,176 [1,167, 1,184] | **1,076** [1,023, 1,133] |

NCM + cache ÷ AR: WMT14 **1.67** [1.50, 1.86], XSum **1.09** [1.04, 1.15].

- **Hypothesis confirmed: NCM loses on CPU.** Relative to AR it moves from 1.27× (GPU) to
  1.67× slower on WMT14, and from a tie (0.94) to 1.09× slower on XSum. With little idle
  parallel hardware, NCM's roughly twice as many processed positions are no longer free.
- **The cache matters more on CPU:** 1.9× on WMT14 and 4.5× on XSum at batch 1, where on
  the GPU it does nothing at batch 1. Without it NCM is 3.2× (WMT14) and 4.9× (XSum) slower
  than AR.
- AR here is eager; compiling it on CPU was not tried.

### Summarization (XSum, 500 test articles)

| | NCM | BART-base (AR) |
|---|---|---|
| parameters (incl. tokenizer) | 149M | 139M |
| ROUGE-1 / 2 / L | 34.7 / 12.9 / **27.8** | 40.9 / 17.9 / **33.5** |
| latency batch 1, median / p95 | 268 / 325 ms | 274 / 354 ms |
| throughput batch 1 / 8 / 32 (examples/s) | 3.7 / 5.0 / 5.0 | 3.7 / 19.6 / **43.0** |
| peak GPU memory batch 32 | 3.4 GB | 0.65 GB |

### Translation (WMT14 De→En, 500 test sentences)

| | NCM | opus-mt-de-en (AR) |
|---|---|---|
| parameters (incl. tokenizer) | 264M | 74M |
| BLEU | 20.0 | **30.8** |
| latency batch 1, median / p95 | **242 / 303 ms** | 283 / 536 ms |
| throughput batch 1 / 8 / 32 (examples/s) | **4.0 / 17.8** / 20.5 | 3.4 / 15.9 / **47.6** |
| peak GPU memory batch 32 | 3.8 GB | 0.24 GB |

![XSum](../results/phase4_xsum_fp16.png)
![WMT14](../results/phase4_wmt14_fp16.png)

### Where NCM's time goes (`scripts/phase4_profile.py`, ms per batch)

| stage | XSum b=1 | XSum b=32 | WMT14 b=1 | WMT14 b=32 |
|---|---|---|---|---|
| level passes (`net.generate`) | 210 | **7599** | 259 | 1446 |
| codes → vectors → token logits | 11 | 40 | 15 | 730 |
| picking tokens (penalties) | 23 | 411 | 0.5 | 35 |

Every level pass re-runs the whole transformer over the full input prefix (512 article
tokens for XSum) because this checkpoint's generation path has no KV cache. XSum does that
9 times. BART encodes the article **once** and then runs ~25 cheap decoder steps with a KV
cache. At batch 1 the GPU is under-used either way, so both take ~270 ms; at batch 32 the GPU
is full and NCM's 9× repeated prefix work dominates.

The official `generate()` does have `use_kv_cache=True` for WMT14. It is 1.5× faster but BLEU
drops from 20.0 to **3.8** on the same 500 sentences, so it is not a valid speedup for this
checkpoint.

### What "31×" means on an RTX 3050

The 31× is NCM against **AURORA-LM, a diffusion LM** that needs 128–184 denoising steps, on an
H100, generating 1024-token outputs at batch 16. NCM needs 8–11 steps, so against diffusion
models the step count alone explains a 10–20× gap. Against an **autoregressive** model of the
same size, which is what most people compare to, on this GPU with short outputs (20 words):

- **batch 1:** NCM is as fast (XSum) or 1.2× faster (WMT14), with a much tighter p95
  (fixed number of passes, independent of output length);
- **batch 32:** NCM is **2.3× (WMT14) to 8.7× (XSum) slower**;
- **quality:** the AR baselines win by 5.7 ROUGE-L and 10.8 BLEU. Caveat: both baselines are
  pretrained models (BART, OPUS data) fine-tuned for the task, while NCM was trained from
  scratch on the task data, so part of this gap is pretraining, not the method. The paper's
  matched-data AR baseline (Table 2) gets ROUGE-L 24.4 (NCM 28.0) and BLEU 25.2 (NCM 22.6).

### Predictions vs reality

| # | prediction | verdict |
|---|---|---|
| 1 | NCM faster at batch 1, 2–5× | **Wrong.** 1.0× (XSum), 1.2× (WMT14). Outputs are only ~20 words, so the AR model needs few steps. |
| 2 | gap shrinks as batch grows | **Right, and stronger than expected:** it flips, AR is 2–9× faster at batch 32. |
| 3 | bigger advantage on XSum (longer outputs) | **Wrong.** XSum is NCM's worst case because its 512-token prefix is re-processed in all 9 passes. |
| 4 | close on XSum, loses on WMT14 | **Half right.** Loses on both against pretrained baselines; matches the paper's own XSum number (27.8 vs 28.0). |
| 5 | NCM uses more memory | **Right:** 3.4–3.8 GB vs 0.2–0.7 GB at batch 32. |
| 6 | fp16: small speedup, same quality | **Right, bigger speedup than expected:** 2–2.5× (Phase 3). |

### Where NCM loses, and where it could win

- Loses: large batches, long inputs (repeated prefix work), short outputs, GPU memory,
  and quality against pretrained AR models.
- Could win: long outputs (an AR model needs one step per token; NCM's step count is fixed),
  latency-sensitive single requests with strict p95 targets, and against diffusion LMs.
  Testing 1024-token generation (the paper's setting) would need the unconditional
  OpenWebText/LM1B checkpoints, which we skipped to save disk space.
