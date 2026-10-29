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

Run the suite, then fill this in from `results/limits_summary.md`. Report every prediction
honestly as right or wrong, as Phase 4 did.
