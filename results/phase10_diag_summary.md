# Phase 10 diagnostic: where the toy NCM loses coherence

512 test stories, 64 tokens. Real stories: GPT-2 perplexity 13.5.

| real levels kept | generated | mode | GPT-2 ppl | token agreement | distinct-2 |
|---|---|---|---|---|---|
| 1..0 | 6 | sampled | 372.3 | 7.5% | 0.594 |
| 1..1 | 5 | sampled | 285.3 | 14.8% | 0.546 |
| 1..2 | 4 | sampled | 248.6 | 20.1% | 0.532 |
| 1..3 | 3 | sampled | 231.0 | 27.3% | 0.522 |
| 1..4 | 2 | sampled | 184.2 | 42.6% | 0.497 |
| 1..5 | 1 | sampled | 111.3 | 67.7% | 0.477 |
| 1..6 | 0 | sampled | 15.5 | 99.0% | 0.423 |
| 1..0 | 6 | greedy | 67.8 | 2.8% | 0.002 |
| 1..1 | 5 | greedy | 52.7 | 18.2% | 0.154 |
| 1..2 | 4 | greedy | 62.8 | 23.4% | 0.274 |
| 1..3 | 3 | greedy | 69.2 | 31.3% | 0.3 |
| 1..4 | 2 | greedy | 74.4 | 49.8% | 0.331 |
| 1..5 | 1 | greedy | 57.8 | 74.9% | 0.365 |
| 1..5 | 1 | last level T=0.7 | 79.7 | 71.6% | 0.427 |
| 1..5 | 1 | last level T=0.5 | 67.3 | 73.4% | 0.394 |
| 1..5 | 1 | last level T=0.3 | 60.9 | 74.4% | 0.374 |
| 1..0 | 6 | last level T=0.7 | 262.1 | 7.6% | 0.526 |
| 1..0 | 6 | last level T=0.5 | 207.3 | 7.7% | 0.471 |
| 1..0 | 6 | last level T=0.3 | 180.8 | 7.7% | 0.436 |
| 1..0 | 6 | all levels T=0.7 | 143.9 | 8.0% | 0.437 |

Teacher-forced (real coarser levels as input):

| level | length | top-1 code accuracy | loss (nats/code) |
|---|---|---|---|
| 1 | 1 | 0.0% | 6.068 |
| 2 | 4 | 9.1% | 5.304 |
| 3 | 8 | 10.7% | 5.042 |
| 4 | 16 | 14.9% | 4.543 |
| 5 | 32 | 21.6% | 3.98 |
| 6 | 64 | 35.8% | 3.121 |
