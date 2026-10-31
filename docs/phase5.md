# Phase 5: NCM inside RAG and agents

Code: `src/ncm_lab/rag/`, `src/ncm_lab/agents/`. Results: `results/phase5_*`.

## Stack

- **LLM:** `qwen2.5:3b` (4-bit, 1.9 GB) through Ollama, installed without sudo into
  `~/.local/ollama` (`scripts/install_ollama_user.sh`, `scripts/ollama.sh serve`).
- **Embeddings:** `all-MiniLM-L6-v2` on CPU. **Vector store:** FAISS inner-product index.
- **Questions:** 40 human-written questions with known answers from XQuAD (the same
  Wikipedia paragraphs and questions exist in English and German). Corpus: its 240 paragraphs.
- **Scoring:** an answer counts as correct if a gold answer appears in it (SQuAD normalization).
- **Agent loop:** written by hand in plain Python (`agents/loop.py`): send messages + tool
  schemas, run requested tools, append results as `tool` messages, repeat.

## Use case 1: compress retrieved chunks with NCM before the LLM

| variant | accuracy | right paragraph retrieved | words sent | total latency |
|---|---|---|---|---|
| plain RAG (top-3 paragraphs) | **87.5%** | 97.5% | 368 | **423 ms** |
| NCM-compressed (XSum summarizer) | 10.0% | 97.5% | 57 | 809 ms |
| AR-compressed (BART-base XSum) | 17.5% | 97.5% | 63 | 792 ms |

Compression **hurts badly**. Retrieval still finds the right paragraph, but a one-sentence,
news-style summary keeps the gist and drops the specific fact the question asks for: it
does not know the question. It is also **slower**: compressing 3 chunks costs ~550 ms, while
the shorter prompt saves the 3B LLM only ~170 ms. Compression could pay off only with a
query-aware compressor, or an LLM where prompt tokens are expensive (a large hosted model).

## Use case 2: German documents, English questions

The 240 German paragraphs (about 1,260 sentences) are translated sentence by sentence, then indexed.

| documents indexed | translation speed (batch 32) | BLEU vs English originals | RAG accuracy |
|---|---|---|---|
| English originals (upper bound) | – | – | 87.5% |
| translated by NCM WMT14 | 18.6 sentences/s (68 s) | 30.8 | 40.0% |
| translated by opus-mt-de-en (AR) | **34.9 sentences/s** (36 s) | **46.6** | **70.0%** |

Re-run on 2026-10-31 with a sentence splitter that no longer splits German ordinals and
abbreviations ("am 3. Oktober", "z. B."); the first run gave 37.5% (NCM) and 67.5% (opus-mt).

Bulk indexing is a large-batch job, which is exactly where NCM is slower (Phase 4). Its
translations also **hallucinate names**: "Luthers Schriften" (Luther's writings) came back
as "Marx's writings", and later in the same paragraph "Blair's career". Entities matter most
for retrieval and answers, hence the 30-point accuracy drop.

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
