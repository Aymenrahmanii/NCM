# Phase 5: NCM inside RAG and agents

> Written with help from Claude (Anthropic). Every number comes from the files in `results/`.

Code: `src/ncm_lab/rag/`, `src/ncm_lab/agents/`. Results: `results/phase5_*`.

## Stack

- **LLM:** `qwen2.5:3b` (4-bit, 1.9 GB) through Ollama, installed without sudo into
  `~/.local/ollama` (`scripts/install_ollama_user.sh`, `scripts/ollama.sh serve`).
- **Embeddings:** `all-MiniLM-L6-v2` on CPU. **Vector store:** FAISS inner-product index.
- **Questions:** all 1,190 human-written questions with known answers from XQuAD (the same
  Wikipedia paragraphs and questions exist in English and German). Corpus: its 240 paragraphs.
  The first run used 40 questions, one per paragraph, and overestimated every accuracy
  (plain RAG 87.5% vs 73.3% on all questions).
- **Scoring:** lenient containment (a gold answer appears in the reply), plus SQuAD exact
  match and token F1, all after SQuAD normalization. Intervals and paired NCM-vs-AR tests:
  `python -m ncm_lab.bench.significance` → `results/significance.md`.
- **Agent loop:** written by hand in plain Python (`agents/loop.py`): send messages + tool
  schemas, run requested tools, append results as `tool` messages, repeat.

## Use case 1: compress retrieved chunks with NCM before the LLM

All 1,190 questions; 95% intervals in brackets.

| variant | contains | exact match | token F1 | right paragraph retrieved | words sent | total latency |
|---|---|---|---|---|---|---|
| plain RAG (top-3 paragraphs) | **73.3%** [70.7, 75.7] | **51.5%** | **0.697** | 96.7% | 377 | **417 ms** |
| NCM-compressed (XSum summarizer) | 6.6% [5.3, 8.1] | 4.3% | 0.095 | 96.7% | 56 | 986 ms |
| AR-compressed (BART-base XSum) | 10.2% [8.6, 12.0] | 6.6% | 0.140 | 96.7% | 64 | 714 ms |

Compression **hurts badly**. Retrieval still finds the right paragraph, but a one-sentence,
news-style summary keeps the gist and drops the specific fact the question asks for: it
does not know the question. NCM is significantly worse than BART on every score
(containment p = 2e-5, exact match p = 0.002, F1 −0.044 [−0.061, −0.028]); at n = 40 that
difference was not significant. It is also **slower**: compressing 3 chunks costs 700 ms
with NCM (uncached generator) and 430 ms with BART, while the shorter prompt saves the 3B
LLM only ~130 ms. Compression could pay off only with a
query-aware compressor, or an LLM where prompt tokens are expensive (a large hosted model).

## Use case 2: German documents, English questions

The 240 German paragraphs (1,265 sentences, with the splitter that keeps "am 3. Oktober" and
"Friedrich II." together; the first run's splitter gave 1,372) are translated sentence by
sentence, then indexed. All 1,190 questions:

| documents indexed | translation speed (batch 32) | BLEU vs English originals | contains | exact match | token F1 |
|---|---|---|---|---|---|
| English originals (upper bound) | – | – | 73.2% [70.6, 75.6] | 51.4% | 0.696 |
| translated by NCM WMT14 | 19 sentences/s (65 s) | 30.8 | 37.9% [35.2, 40.7] | 28.3% | 0.465 |
| translated by opus-mt-de-en (AR) | **37 sentences/s** (34 s) | **46.6** | **55.8%** [53.0, 58.6] | **41.2%** | **0.592** |

NCM vs opus-mt on the same questions: containment p = 5e-35 (267 questions only opus-mt gets
right, 54 only NCM), exact match p = 9e-21, F1 −0.127 [−0.150, −0.104].

The translation speed was measured with the uncached official generator. With the corrected KV
cache (Phase 4, upstream PR #5), NCM's WMT14 batch-32 throughput matches opus-mt, so the 2× is
an implementation gap. The quality gap is not: NCM translations **hallucinate names**.
"Luthers Schriften" (Luther's writings) came back as "Marx's writings", and later in the same
paragraph "Blair's career". Entities matter most for retrieval and answers, hence the
18-point accuracy drop.

## Use case 3: agent with NCM tools

Tools: `search_docs`, `summarize_fast` (NCM XSum), `translate_de_en` (NCM WMT14),
`solve_sudoku` (NCM Sudoku-hard). Six multi-step tasks, every step logged with its latency
(`results/phase5_agent_log.jsonl`).

| | first system prompt | stricter prompt with explicit tool rules |
|---|---|---|
| graded answers correct | 1 / 4 | **3 / 4** |
| tasks that called no tool | 2 (lookup, sudoku) | 0 |
| total time for the 6 tasks | 64 s | **21 s** |
| share of time: LLM / NCM tools | 97% / 3% | 91% / 9% |

What went wrong, and why:
- With the first prompt the LLM answered a lookup question from memory (wrong) and spent
  28 s "solving" the sudoku itself in 1,449 tokens instead of calling `solve_sudoku`.
  Explicit rules ("a sudoku: call solve_sudoku, never solve it yourself") fixed both.
- The sudoku tool was then called with **82 digits** for an 81-cell puzzle: small LLMs
  cannot copy long digit strings reliably. The tool rejected it, as it should.
- The LLM sent the **German** text to `summarize_fast` instead of the English translation;
  the English-only summarizer returned garbage.

**Speed:** even a perfectly fast NCM would make this agent at most 3–9% faster. The LLM's
reasoning and tool-call generation is the bottleneck, not the specialist tools.

## Where NCM helped, where it hurt, and why

- **Helped:** a fixed, predictable latency per call (the same number of passes whatever the
  output); the Sudoku model is genuinely strong (84.5% of hard puzzles) and fast.
- **Hurt:** compression removed the facts RAG needs; translation was slower at batch size
  and less accurate than a small pretrained AR model; inside an agent its speed is
  irrelevant because the LLM dominates.
- **Why:** the released checkpoints are small task-specific models trained from scratch on
  one dataset (news summaries, WMT14), while the baselines are pretrained. NCM's speed
  advantage only appears at small batch sizes and long outputs, which none of these
  pipelines needed.
