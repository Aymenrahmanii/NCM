# Phase 3: the official NCM checkpoints

Code: `src/ncm_lab/models/ncm.py`. Results: `results/phase3_eval.md`, `results/phase3_eval.csv`.

```python
from ncm_lab.models.ncm import NCMSummarizer, NCMTranslator, NCMSudokuSolver
NCMSummarizer().generate(["Long news article ..."])
NCMTranslator().generate(["Die Katze schläft auf dem Sofa."])
NCMSudokuSolver("hard").generate(["530070000600195000..."])  # 81 cells, 0 = empty
```

## Checks against the official code (same 50 test examples)

| | my wrapper | official `evaluate.py` | paper (full test set) |
|---|---|---|---|
| XSum ROUGE-L | 26.47 | 26.48 | 28.0 |
| WMT14 BLEU | 15.28 | 15.28 | 22.6 |
| Sudoku solved, easy / medium / hard (200 each) | – | 99.0 / 97.5 / 84.5 % | 99.5 / 96.5 / 86.1 % |

On 500 examples (Phase 4) the scores move to ROUGE-L 27.8 and BLEU 20.0: the first 50 WMT14
sentences all come from one hard local news article.

## Precision and memory (50 examples)

| task | precision | weights | peak memory | ms / example | score |
|---|---|---|---|---|---|
| XSum | fp32 | 691 MB | 1498 MB | 436 | R-L 26.47 |
| XSum | fp16 | 698 MB | 1694 MB | 174 | R-L 26.55 |
| WMT14 | fp32 | 1086 MB | 1831 MB | 109 | BLEU 15.28 |
| WMT14 | fp16 | 1086 MB | 2454 MB | 59 | BLEU 14.96 |

fp16 is 2–2.5× faster with the same quality. My "fp16" is `torch.autocast`: weights stay in
fp32 and fp16 copies are made on the fly, so peak memory goes *up*. Casting the weights
themselves to fp16 would halve the weight memory.

## What surprised me compared with my toy version

1. **The tokenizer is as big as the generator.** XSum: 88M generator + 61M VQ-VAE.
   WMT14: 120M + 144M. The paper reports only the generator size.
2. **Conditioning is simple.** For XSum and WMT14 the input text is not quantized at all:
   raw GPT-2 / Qwen3 token ids go through the tokenizer's embedding table and act as a
   prefix the generator attends to (`raw_token_prefix: true`).
3. **Turning my own text into model input** = encode with the same Hugging Face tokenizer
   named in `config.yaml` (GPT-2 for XSum, Qwen3 for WMT14), cut to `prefix_len`
   (512 / 96) and pad with the configured pad id. Verified: raw text gives exactly the same
   ROUGE as the pre-tokenized eval data.
4. **The eval file hides the answer.** Each XSum item is 560 ids: the article padded to
   512, then the 48-token gold summary. My first raw-text test decoded all 560 ids, the
   padding vanished, the summary slid into the model's input, and ROUGE jumped from 26.5 to
   28.7. A leak like that would have made every later comparison wrong.
5. **Decoding is not plain argmax.** XSum uses a frequency penalty against the training
   unigram distribution plus an adjacency penalty, applied position by position. That loop
   matters for speed (Phase 4).
6. **More levels than my toy:** 9 levels for XSum (4 → 48 positions), 11 for WMT14
   (1 → 64), codebooks of 4096, RoPE and QK-norm in the generator.
7. **Limits of the models:** XSum reads at most 512 GPT-2 tokens and writes at most 48;
   WMT14 reads 96 Qwen tokens and writes 64, so long German text must be split into
   sentences first.

## Config settings, explained

- `levels.manual_levels`: length of each level, coarse to fine (e.g. 4, 6, 10, … 48).
- `vqvae_vocab_size`: codebook size per level (4096).
- `prefix_len`: how many input tokens the generator can read (512 XSum, 96 WMT14).
- `context_length`: prefix + output length (560 = 512 + 48).
- `generation.temperature / top_k`: `top_k: 1` means greedy, one pick per code.
- `frequency_penalty`, `adjacency_penalty`: discourage repeated words in the final tokens.
- `cfg_scale`: classifier-free guidance strength (0 = off; doubles the batch when on).
- `fine_drop.prob: 0.5`: nested dropout, as in my toy tokenizer.
