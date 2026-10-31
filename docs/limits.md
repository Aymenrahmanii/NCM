# Limits suite: where NCM wins and loses

> Draft written with Claude. The predictions below are suggestions: **edit them into your own,
> then commit before running anything**, so the git history shows they came first.

Phases 3–5 answered "is NCM faster, at what quality, and does it help RAG and agents?" on one
operating point each. This suite pushes each axis until something breaks, on the same
RTX 3050 6 GB, and adds a general-purpose LLM as a third point of comparison.

```bash
python -m ncm_lab.limits.run --list                  # the eight experiments
python -m ncm_lab.limits.run faithfulness            # CPU only, needs Phase 4 outputs
python -m ncm_lab.limits.run gpu-only                # everything that does not need Ollama
bash scripts/ollama.sh serve &                       # then the LLM experiments:
python -m ncm_lab.limits.run sudoku-stress llm-vs-ncm agent-reliability
python -m ncm_lab.limits.report                      # summary + charts + LaTeX tables
```

Outputs: `results/limits_<name>.csv` (one row per condition), `results/limits_<name>_details.jsonl`
(sample outputs), `results/limits_summary.md` and `results/limits_*.png`, and
`paper/tables/limits_*.tex` for the paper. Every experiment also runs on CPU in the tests
with tiny stand-in models (`tests/test_limits.py`), so the code paths are checked before a
GPU hour is spent.

## The experiments

| # | name | question | design | fairness notes |
|---|---|---|---|---|
| 1 | `input-length` | Where does NCM's fixed input window break translation? | WMT14 test sentences joined k = 1, 2, 3, 4, 6, 8 at a time; BLEU, chrF, length ratio, share of inputs longer than the window | AR gets `max_new_tokens=400` so its usual 64-token cap is not the limit measured |
| 2 | `batch-sweep` | At which batch size does AR overtake NCM, and what fits in 6 GB? | batch 1 … 128, 128+ examples per size, until out-of-memory | warm-up per shape; generation only, as in Phase 4 |
| 3 | `output-length` | How long must outputs be before NCM's fixed step count wins? | AR forced to write exactly L = 16 … 400 tokens; NCM's constant cost; fitted break-even L | speed only: forced AR text is not meaningful, and NCM cannot exceed its canvas |
| 4 | `faithfulness` | How often do outputs invent numbers, names, or repeat? | Phase 4's 500 outputs per model against their sources | simple string proxies, same for both models: compare models, not absolute rates |
| 5 | `robustness` | How brittle is each model to messy input? | typos (10%, 30% of words), lowercase, uppercase, no punctuation | identical perturbations (seeded) for both models |
| 6 | `sudoku-stress` | Does NCM generalize, and can it say "no solution"? | hard, shuffled (relabelled + transposed), 5 / 15 clues removed, unsolvable, already solved; vs an exact solver and qwen2.5:3b | any valid completion counts for under-specified puzzles |
| 7 | `llm-vs-ncm` | 150–260M specialist vs 3B general LLM on the same tasks | XSum and WMT14, 100 examples, one request at a time, end-to-end latency | LLM is zero-shot and 4-bit quantized; specialists were trained on these tasks |
| 8 | `agent-reliability` | How reliably does the LLM drive the NCM tools? | the six Phase 5 tasks, 10 runs at temperature 0.7 + 1 at 0 | counts right tools, tool errors, step-limit hits, accuracy |

## Predictions (edit, then commit before running)

1. **Input length:** NCM's BLEU falls sharply once inputs pass its window (k ≥ 4, ~80+
   words) and its outputs get relatively shorter; opus-mt loses only a few BLEU.
2. **Batch sweep:** AR overtakes NCM at batch 2–4 on XSum and 8–16 on WMT14; NCM runs
   out of memory first (around batch 64 on XSum).
3. **Output length:** AR becomes slower than NCM at around 30–60 output tokens at batch 1.
   That is NCM's real speed niche, but these checkpoints cannot use it.
4. **Faithfulness:** NCM invents numbers and names noticeably more often than the AR
   baselines and repeats words several times more often.
5. **Robustness:** both lose quality under uppercase and heavy typos; NCM loses more,
   because it was trained on far less data.
6. **Sudoku:** NCM stays near 84% on shuffled puzzles (it learned the rules, not the
   puzzles), drops on sparser puzzles, and returns a confident full grid for 100% of the
   unsolvable ones. qwen2.5:3b solves almost none.
7. **Specialist vs LLM:** NCM is 3–10× faster than qwen2.5:3b per request and wins on ROUGE
   (the LLM's summaries are not XSum-style); the LLM wins on translation BLEU.
8. **Agent:** at temperature 0.7 the right tools are called in 70–90% of runs; most
   failures are skipped tool calls rather than malformed arguments.

## Results (RTX 3050 6 GB, fp16; run on 2026-10-30)

Full tables: [`results/limits_summary.md`](../results/limits_summary.md). Charts:
`results/limits_*.png`. `llm-vs-ncm` and `agent-reliability` did not run (Ollama was not
up), and `sudoku-stress` ran without its LLM rows.

| # | prediction | verdict |
|---|---|---|
| 1 | NCM's BLEU collapses past its 96-token input window (k ≥ 4); opus-mt loses a few BLEU | **Right direction, wrong cause, and earlier.** NCM 16.3 → 10.1 → 2.6 → 0.04 BLEU (k = 1, 2, 3, 8) while 85% of k = 2 inputs still fit the window. The limit is the **output canvas**: NCM writes 18–25 words whatever the input (length ratio 0.91 → 0.15). opus-mt *improves* (24.7 → 31.5). |
| 2 | AR overtakes at batch 2–4 (XSum), 8–16 (WMT14); NCM runs out of memory first | **Half right.** WMT14: opus-mt wins from batch 16 for good (NCM faster at 2–8). XSum: BART is faster at every batch size; NCM's throughput peaks at batch 16 and falls. **No OOM**: under WSL2 the driver spills past 6 GB into system RAM (NCM peak 9.4 GB) and throughput collapses instead. |
| 3 | AR slower than NCM beyond 30–60 output tokens | **Wrong range, right idea.** Break-even at 20 (XSum) and 26 (WMT14) tokens at batch 1, but NCM's canvas ends at 48 / 64 tokens: its speed niche is 20–64 tokens, worth ≤ 2.3×. At batch 8 on XSum there is no niche at all. |
| 4 | NCM invents numbers/names more and repeats far more | **Right.** Repetition 37.8% vs 5.4% (XSum), 36.4% vs 13.4% (WMT14); invented numbers 20.2% vs 11.4% (XSum); invented names 21.4% vs 7.4% (WMT14). The name proxy is uninformative on XSum (56% vs 51%). |
| 5 | NCM loses more quality under noise | **Mostly wrong.** Absolute drops are similar; uppercase destroys both (NCM WMT14 BLEU 18.8 → 0.3, opus-mt 27.5 → 2.2). Relative to its lower start, NCM loses somewhat more. |
| 6 | NCM ~84% on shuffled, drops on sparser, always answers unsolvable | **Right, and worse than predicted.** 84% hard, 85% shuffled (symmetry-invariant), but **32%** with 5 clues removed and **1%** with 15; full grid returned for 100% of unsolvable puzzles; **0% on already-solved grids** (it overwrites 7–10 given digits). The exact CPU solver: 100% in 0.6 ms. |
| 7 | NCM 3–10× faster than qwen2.5:3b; LLM wins on BLEU | not run yet |
| 8 | right tools in 70–90% of sampled agent runs | not run yet |

Surprises worth following up:
- The hard test puzzles inspected all have exactly 51 blanks. NCM may have learned a fixed
  amount of change rather than "keep the clues"; check the blank count over all puzzles.
- Batch-1 times moved by up to 25% between this run and Phase 4 (e.g. opus-mt 203 vs 283 ms):
  on a WSL laptop, batch-1 differences of that size are noise, so read batch 1 as a tie.
