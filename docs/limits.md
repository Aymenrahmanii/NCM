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

## Results

From `results/limits_summary.md` (run 2026-11, RTX 3050 6 GB laptop, fp16 autocast). The
suite ran **before** the KV-cache fix, so every NCM number here is the official uncached
generator with fp32 weights; the throughput and memory rows would look better today (see
`docs/phase4-results.md`). Samples are 100–500 examples per condition, without intervals, so
treat differences of a point or two as noise.

| # | Prediction | Verdict | What happened |
|---|---|---|---|
| 1 | Input length: NCM falls sharply from k ≥ 4; opus-mt loses a few BLEU | **Partly right** | NCM breaks earlier than predicted: BLEU 16.3 → 10.1 at k = 2 and 2.6 at k = 3, as 15% then 54% of inputs overflow its prefix window and the output length ratio falls from 0.91 to 0.37. opus-mt did not lose BLEU; it *gained* (24.7 → 31.5 at k = 6, 29.7 at k = 8) with a steady length ratio of about 0.9. Corpus BLEU on longer segments is not directly comparable to sentence-level BLEU, so read this as "no loss". |
| 2 | Batch sweep: AR overtakes at batch 2–4 (XSum), 8–16 (WMT14); NCM OOMs first, ~64 on XSum | **Right on XSum, wrong on WMT14 and on OOM** | XSum: BART is ahead from batch 2. WMT14: opus-mt is faster at batch 1, NCM at 2 and 4 (18.2 vs 11.6 ex/s), tied at 8, and opus-mt ahead from 16 (9× at 128). Nothing raised OOM: at batch 128 NCM's peak (7.8 GB XSum, 9.4 GB WMT14) exceeds the 6 GB card and the driver falls back to shared system memory. On XSum that makes each example 4.8× slower than at batch 64; on WMT14 throughput holds. |
| 3 | Output length: AR slower than NCM beyond ~30–60 tokens at batch 1 | **Right** | Break-even 27 tokens (XSum) and 31 (WMT14) at batch 1; at batch 8, 120 and 39. The canvases (48 and 64 tokens) are longer than the batch-1 break-even, but typical outputs (~20 words) are not, which explains the batch-1 tie. At batch 8 on XSum the break-even is past the canvas, so the uncached generator cannot win there at any length. |
| 4 | Faithfulness: NCM invents more numbers and names, repeats several times more | **Right** | Invented numbers 20% vs 11% (XSum) and 3.8% vs 0% (WMT14); invented names 56% vs 51% (XSum, small gap on a noisy proxy) and 21% vs 7% (WMT14); repetition 38% vs 5% (XSum, 7×) and 36% vs 13% (WMT14). |
| 5 | Robustness: both lose under uppercase and heavy typos; NCM loses more | **Mostly wrong** | Absolute drops are about the same for both models (WMT14 typos 10%: −3.0 vs −2.75 BLEU; no punctuation −6.9 vs −6.9). Relative to its lower clean score NCM loses more, and two cases are clearly worse for NCM: missing punctuation on XSum (−6.3 vs −3.4 ROUGE-L) and uppercase (NCM keeps 19% of its ROUGE-L, BART 40%). Uppercase breaks both. |
| 6 | Sudoku: ~84% on shuffled; drops on sparser; confident grid on 100% of unsolvable; LLM ~0 | **Right** | 85% shuffled vs 84% original; 32% with 5 clues removed, 1% with 15; a full grid for 100% of unsolvable puzzles and never a refusal. Unpredicted: NCM returns a *valid* grid for **0%** of already-solved puzzles, so it cannot even copy a complete grid it was never shown in training. qwen2.5:3b solves none except already-solved grids (100%). |
| 7 | Specialist vs LLM: NCM 3–10× faster; wins ROUGE, loses BLEU | **Right, except speed on WMT14** | Median latency 239 vs 831 ms (3.5×) on XSum but 250 vs 501 ms (2.0×) on WMT14. ROUGE-L 27.6 vs 19.7 for NCM; BLEU 16.2 vs 21.9 for the LLM. The pretrained AR baselines beat both on quality. The LLM runs in a different runtime (Ollama), so its latency is indicative only. |
| 8 | Agent: right tools in 70–90% of runs; failures mostly skipped calls | **Wrong on the rate, half right on the cause** | Tool choice is not random but fixed per task: 100% right on four tasks and 0% on two, at temperature 0 and 0.7 alike. On those two the LLM skips a tool every run (no summarize step; translates but never searches, then invents an answer). On Sudoku it calls the right tool but passes a malformed 82-cell puzzle in 10 of 11 runs and then gives up. |

**What this changes in the plan.** Prediction 3 is the strongest result: NCM's speed niche on
this GPU begins at about 30 output tokens at batch 1, so Phase 9 (long outputs) is the right
next test of the speed claim. Predictions 2 and 4 should be re-run with the cached, fp16-weight
generator before they go in the paper as current numbers.
