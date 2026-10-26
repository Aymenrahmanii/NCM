# Phase 2 results (toy NCM vs autoregressive, RTX 3050)

Params: tokenizer 8.57M, NCM generator 5.55M, AR 6.86M

## Reconstruction accuracy by levels used

| levels | length of last level | accuracy |
|---|---|---|
| 1 | 1 | 17.0% |
| 2 | 4 | 20.7% |
| 3 | 8 | 28.2% |
| 4 | 16 | 45.1% |
| 5 | 32 | 73.1% |
| 6 | 64 | 98.7% |

## Sample quality (512 samples, temperature 1.0)

| source | GPT-2 perplexity (lower = more fluent) | distinct-2 |
|---|---|---|
| real test stories | 13.2 | 0.419 |
| toy NCM | 377.2 | 0.590 |
| AR baseline | 24.5 | 0.472 |

## Generation time (64 tokens, median of 10 runs)

| model | batch | ms per batch | samples/s |
|---|---|---|---|
| toy NCM (6 passes) | 1 | 33.89 | 29.5 |
| AR + KV cache (64 passes) | 1 | 316.32 | 3.2 |
| AR no cache | 1 | 342.17 | 2.9 |
| toy NCM (6 passes) | 16 | 37.2 | 430.1 |
| AR + KV cache (64 passes) | 16 | 287.06 | 55.7 |
| AR no cache | 16 | 312.94 | 51.1 |
| toy NCM (6 passes) | 128 | 88.82 | 1441.2 |
| AR + KV cache (64 passes) | 128 | 269.03 | 475.8 |
| AR no cache | 128 | 673.11 | 190.2 |

Batch 1: toy NCM is 9.3x faster than AR with KV cache.

Batch 16: toy NCM is 7.7x faster than AR with KV cache.

Batch 128: toy NCM is 3.0x faster than AR with KV cache.
