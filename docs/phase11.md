# Phase 11: does the generator need full attention?

> Written with help from Claude (Anthropic). Every number comes from the files in `results/`.

**Status: exploratory.** Phase 10's kill rule skips this phase, because the generator does not
yet write fluent text. It ran anyway, as the question "does the generator need full
attention?", not as a route to a competitive NCM. The plan's re-runs of the top two variants
at 200M tokens were replaced by a validity check of the Phase 10 reimplementation (see
`docs/phase10.md`), which mattered more.

## Setup (2026-11-16)

```bash
bash scripts/phase10_train.sh phase11                     # 4 generators, 100M positions each
python -m ncm_lab.scale.evaluate quality --tag _phase11 --runs gen25m-attn-100m_s0 ...
python -m ncm_lab.scale.mixer_speed --device cuda --repeats 20
python -m ncm_lab.scale.mixer_speed --device cpu --repeats 5
```

- Frozen Phase 10 tokenizer, gen25m width and depth, 100M positions (half the Phase 10
  budget), seed 0, everything else as in Phase 10. Code: `src/ncm_lab/scale/models.py`.
- **attn:** the Phase 10 generator, retrained at 100M as the baseline.
- **nocross:** each level attends only to itself. Its input already holds the canvas of all
  coarser levels, so cross-level attention might be redundant. A mask change only.
- **conv:** depthwise convolution (kernel 7) on levels of 64 positions or more, attention on
  the coarse levels.
- **linear:** linear attention (elu + 1 feature map), block-causal across levels.
- Mamba (variant 4) was not run: the plan allows it only if conv and linear both fail on
  speed, and conv did not.
- Speed: one full sample (every level) at batch 1, random weights, at 256, 1024 and 2048
  positions; median of 20 runs on the RTX 3050, 5 on the CPU.

## Results

Quality (`results/phase10_quality_phase11.csv`): temperature tuned on validation as in
Phase 10, 512 test samples, MAUVE against 5 reference draws. Training loss is the
teacher-forced loss per code at the end of training, on the three finest levels.

| mixer | params | train min | MAUVE ↑ | GPT-2-L ppl | loss level 7 | level 8 | level 9 |
|---|---|---|---|---|---|---|---|
| human test stories | – | – | 0.952 | 7.1 | – | – | – |
| attn | 24.6M | 77 | 0.0051 | 372 | **4.52** | **3.75** | **2.51** |
| nocross | 24.6M | 76 | 0.0056 | 132 | 4.63 | 3.82 | 2.56 |
| conv | 26.5M | 61 | 0.0070 | 111 | 4.86 | 4.02 | 2.77 |
| linear | 24.6M | 147 | 0.0052 | 209 | 4.61 | 3.93 | 2.76 |

Losses and minutes come from each run's `artifacts/scale/gen25m-<mixer>-100m_s0/log.csv`.

Speed in ms per sample (`results/phase11_speed_cuda.csv`, `_cpu.csv`):

| mixer | GPU 256 | GPU 1024 | GPU 2048 | CPU 256 | CPU 1024 | CPU 2048 |
|---|---|---|---|---|---|---|
| attn | **58** | **86** | 148 | 243 | 957 | 2,360 |
| nocross | 63 | 88 | 154 | 249 | 941 | 2,403 |
| conv | 75 | 103 | **130** | **227** | **631** | **1,153** |
| linear | 194 | 279 | 337 | 4,619 | 23,991 | 39,684 |

- **MAUVE cannot rank the mixers.** All four sit at the Phase 10 floor (0.005–0.007), so
  "quality within seed noise of the baseline" is met trivially and means nothing. The
  training loss is the only usable quality signal, and it favours full attention at every
  fine level.
- **Cross-level attention is nearly redundant:** removing it costs 0.05 nats at the finest
  level and saves no time (the mask does not shrink the work in this implementation).
- **Convolution is the only speed win, and only on the CPU:** 1.5× at 1024 positions and
  2.0× at 2048. On the GPU it is slower up to 1024 and 1.14× faster at 2048, short of the
  1.3× target. It also has the worst fine-level loss (+0.26 nats at level 9, about the same as
  linear attention).
- **Linear attention is slow because of this implementation,** which computes one level at
  a time to stay within memory; it is not a measurement of linear attention in general.

## Verdict

Against the plan's criteria: conv meets the speed target on the CPU (1.5× at 1024) but not on
the GPU, and its quality cannot be judged by MAUVE while every generator fails; its training
loss is worse. The kill rule ("no variant 1.2× faster at any length up to 1024") is not
triggered, because of conv on the CPU.

The honest summary: **at this scale attention is not what limits the NCM, and the only cheaper
mixer that is faster (convolution, on the CPU) learns the fine levels worse.** The question
is worth reopening only with a generator that writes fluent text, which Phase 10's validity
check shows this one does not, with either tokenizer.

## Caveats

- One seed, half the Phase 10 budget; a slow-converging variant could catch up at 200M
  (the plan's re-runs were not done).
- Speed uses random weights and the Phase 10 level schedule; the 2048-position runs scale
  that schedule up.
- The losses compare mixers on the same tokenizer and data, so they are comparable to each
  other but not to the validity-check generator, which uses a different tokenizer.
