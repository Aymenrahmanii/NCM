# Phase 12: adapting the released NCM to a narrow task (SAMSum)

**Question:** what is the cheapest way to adapt a released NCM checkpoint to a new
conditional task? Phase 10 stopped scaling NCM for open-ended text; conditional tasks are
where the released models work, so this phase stays with them.

## Setup (2026-11-15)

- **Task:** SAMSum dialogue summarization, 14,731 train / 818 validation / 819 test. The
  official `Samsung/samsum` repository is no longer public; the data comes from the
  `knkarthick/samsum` CSV mirror. Validation and test match the published sizes exactly;
  train has one row fewer, the empty dialogue other copies also drop. Licence CC BY-NC-ND
  4.0, so the data stays out of git.
- **Models:** the released XSum NCM (88M generator, 61M VQ tokenizer, prompt of 512 GPT-2
  tokens, summary canvas of 48) and `morenolq/bart-base-xsum` (140M), the Phase 4 baseline,
  which starts from the same XSum-tuned point.
- **Arms:** zero-shot; LoRA rank 8 on query and value; LoRA rank 16 on every attention and
  MLP projection; full fine-tuning (the NCM's VQ tokenizer stays frozen). LoRA is written by
  hand (`src/ncm_lab/finetune/lora.py`, alpha = 2 × rank); on the NCM's fused `c_attn` the
  "query and value" arm masks the key columns.
- **Training, fixed in advance:** 3 epochs, batch 16, AdamW (0.9, 0.98, wd 0.01), lr 3e-4
  for LoRA and 3e-5 for full, 5% warm-up then cosine, bf16 autocast, clip 1.0, seeds 0–2.
  No early stopping; validation loss is logged per epoch.
- **Decoding:** each model's own setting: the NCM's released config (greedy codes, frequency
  penalty), BART greedy with 64 new tokens, as in Phase 4.
- **Metrics on all 819 test dialogues:** ROUGE-1/2/L (stemmed), and the Phase 7 faithfulness
  proxies against the dialogue (invented numbers, invented names, repetition).

## Gate: can the frozen tokenizer write SAMSum summaries?

`scripts/phase12_gate.py`, `results/phase12_gate.json`. Each summary is closed with `<eos>`
and padded to 48 tokens, encoded to codes and decoded back:

| split | token accuracy | rebuilt exactly | longer than 47 tokens (cut) |
|---|---|---|---|
| validation | 98.97% | 77.9% | 63 of 818 (7.7%) |
| test | 98.94% | 76.9% | 57 of 819 (7.0%) |

It passes the 95% gate, so no decoder fine-tuning arm is needed. The cut summaries cap the
NCM's ROUGE slightly; BART has no such limit.

## Rebuilding the training loss

The official repository ships inference code only. The XSum generator uses block-diagonal
attention: each level sees the prompt and itself, and its input is the canvas of all coarser
levels. So one teacher-forced pass with the real codes as inputs gives every level's logits,
the same ones `generate` computes level by level (`src/ncm_lab/finetune/ncm_loss.py`).
Checked on the released weights before training (`results/phase12_loss_check.json`, 64
validation dialogues):

| check | result |
|---|---|
| teacher-forced argmax = greedy generation's codes | **100.0%** agreement |
| loss on the model's own greedy codes | 1.43 |
| loss on the real SAMSum summaries (zero-shot start) | 3.19 |
| loss on summaries paired with the wrong dialogue | 3.57 |

The rebuild took about half an hour, well inside the plan's 6-hour limit, so the phase stays
on the official checkpoint rather than falling back to the Phase 10 model.

Cost on the RTX 3050: NCM full fine-tuning at micro-batch 8 reserved 5.0 GB and spilled
(the 512-position prompt is always full length), so the NCM uses micro-batch 4 (3.6 GB,
1.4 s per step of 16). About 1 h per NCM run and 25 min per BART run, 13–14 h in total.

## Results (2026-11-16)

All 819 test dialogues, 3 seeds per trained arm (zero-shot is deterministic, one run).
`results/phase12_runs.csv` (per run), `results/phase12_summary.csv` (means and SDs),
`results/phase12_lora_vs_full.csv` (paired bootstrap), outputs in `results/phase12_outputs/`.

| model | arm | trainable | ROUGE-1 | ROUGE-2 | ROUGE-L (± SD) | min / run |
|---|---|---|---|---|---|---|
| NCM | zero-shot | – | 12.5 | 1.0 | 9.9 | – |
| NCM | LoRA r8, query + value | 0.29M | 27.7 | 6.9 | 22.12 ± 0.14 | 69 |
| NCM | LoRA r16, all layers | 2.4M | 31.9 | 9.0 | 24.92 ± 0.36 | 76 |
| NCM | full | 88M | 33.0 | 9.7 | **25.83** ± 0.14 | 79 |
| BART | zero-shot | – | 20.3 | 2.7 | 15.4 | – |
| BART | LoRA r8, query + value | 0.44M | 44.9 | 19.5 | 36.47 ± 0.11 | 15 |
| BART | LoRA r16, all layers | 3.2M | 48.0 | 22.6 | 39.30 ± 0.20 | 19 |
| BART | full | 139M | 48.2 | 23.1 | **39.71** ± 0.17 | 23 |

**Success criterion: a LoRA arm within 1 ROUGE-L of full fine-tuning.** Paired bootstrap over
the 819 dialogues, each dialogue's ROUGE-L averaged over the 3 seeds:

| model | LoRA arm − full | 95% CI | met? |
|---|---|---|---|
| BART | r16 all: −0.41 | [−0.94, +0.12] | **yes**, the whole interval is above −1 |
| NCM | r16 all: −0.90 | [−1.34, −0.46] | on the point estimate only; the interval reaches −1.34 |
| BART | r8 qv: −3.24 | [−4.00, −2.48] | no |
| NCM | r8 qv: −3.71 | [−4.28, −3.15] | no |

Faithfulness proxies against the dialogue (share of summaries; lenient name check, which
flags 2.9% of the human references):

| model | arm | invented number | invented name | repetition | words |
|---|---|---|---|---|---|
| human references | – | 1.2% | 2.9% | 14.7% | 20.0 |
| NCM | zero-shot | 23.3% | 72.4% | 50.8% | 18.0 |
| NCM | full | 7.1% | 36.3% | 52.9% | 18.7 |
| BART | zero-shot | 2.9% | 53.6% | 15.8% | 16.7 |
| BART | full | 0.4% | 2.9% | 15.9% | 17.0 |

- **Adaptation works for both, and rank-16 LoRA on every projection is the cheap option:**
  under 3% of the weights, within 0.4 (BART) to 0.9 (NCM) ROUGE-L of full fine-tuning.
  Query-and-value LoRA at rank 8, the common default, is 3–4 points short for both.
- **It does not close the gap between the models.** Fine-tuned, NCM stays 14 ROUGE-L behind
  BART (25.8 vs 39.7), and the faithfulness gap barely moves: NCM still invents a name in a
  third of its summaries (BART: 2.9%, the human rate) and repeats itself in half (BART: 16%,
  near the human 15%).
  The summaries pick up the right people and topic but not the sentences ("Am's number has
  texted Betty.. grandmother her last time."), the finest-level failure Phase 10 located.
- **LoRA saves memory, not time, on this GPU.** NCM runs take 69–79 min whatever the arm:
  the forward and backward pass through the frozen 512-position prompt dominates.
- **BART is pretrained and the NCM is not**, so the model gap is not a clean method
  comparison (as the plan noted); the adaptation question, LoRA vs full on the same starting
  point, is clean for each model.
