# Phase 4: where NCM spends its time (fp16, ms per batch)

## xsum

| stage | batch 1 | batch 32 |
|---|---|---|
| level passes (net.generate) | 209.7 | 7598.8 |
| codes -> vectors -> logits | 10.5 | 39.5 |
| pick tokens (penalties) | 22.6 | 411.0 |
| **total** | 242.7 | 8049.4 |

## wmt14

| stage | batch 1 | batch 32 |
|---|---|---|
| level passes (net.generate) | 258.6 | 1445.8 |
| codes -> vectors -> logits | 14.8 | 729.7 |
| pick tokens (penalties) | 0.5 | 34.7 |
| **total** | 274.0 | 2210.2 |

