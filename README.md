# NCM Lab: Next-Concept Models on a Laptop GPU

[![CI](https://github.com/Aymenrahmanii/NCM/actions/workflows/ci.yml/badge.svg)](https://github.com/Aymenrahmanii/NCM/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)
![GPU: RTX 3050 6 GB](https://img.shields.io/badge/GPU-RTX%203050%206%20GB-76b900)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An independent study of **Next-Concept Modeling (NCM)**, a new text generation method from
Hazy Research ([paper](https://arxiv.org/abs/2610.14162), [code](https://github.com/HazyResearch/ncm))
that writes text coarse to fine instead of one token at a time and is reported to be up to
**31× faster**. This project asks what that speed means on an ordinary laptop GPU, at what
quality, and whether it helps inside real RAG and agent pipelines.

It goes from a toy NCM built from scratch to the official checkpoints, a fair benchmark
against autoregressive models and a 3B LLM, a FastAPI service, and a suite of 8 experiments
that push NCM until it breaks. Every number is reproducible from this repository.

**📄 Report (PDF): [paper/ncm-limits-report.pdf](paper/ncm-limits-report.pdf)**

*Aymen Rahmani, final-year Data Science Engineering student. Personal learning project.*

![NCM's BLEU collapses as inputs grow, while an autoregressive translator improves](results/limits_input_length.png)

## Key findings

| question | answer (RTX 3050 6 GB) |
|---|---|
| What is the "31×" measured against? | A **diffusion** language model (128+ steps) on an H100 with 1024-token outputs, not the GPT-style autoregressive (AR) models most people use. |
| Is NCM faster than a same-size AR model? | **At batch 1, roughly a tie** (each wins within ±25% between runs). At batch 32, **AR is 2–10× faster**. NCM wins clearly only for single requests with **20–64-token outputs** (≤ 2.3×). |
| At what quality? | It reproduces the paper (XSum ROUGE-L 27.8 vs 28.0; Sudoku-hard 84.5% vs 86.1%) but trails pretrained AR models: ROUGE-L 27.8 vs 33.5, BLEU 20.0 vs 30.8. |
| Where does it break? | Its fixed output canvas caps every output at about 23 words, so translation BLEU falls from 16.3 to 0.04 as inputs grow. It repeats itself in ~37% of outputs (AR: 5–13%). Sudoku drops from 84% to 32% with five clues removed, and it "solves" unsolvable puzzles. |
| vs a 3B general LLM (qwen2.5:3b)? | NCM is 3.7× faster and better at news summaries (ROUGE-L 27.6 vs 20.2) but worse at translation (BLEU 16.2 vs 21.9). Small trained AR models beat both. |
| Does speed matter in RAG and agents? | **No.** NCM context compression cut RAG accuracy from 87.5% to 10%; NCM translation gave 40% vs 70% for a small AR translator; in the agent the LLM takes 91–97% of the time and makes the same planning mistakes in 11/11 runs. |

Full write-ups: [Phase 2](docs/phase2.md) · [Phase 3](docs/phase3.md) ·
[Phase 4](docs/phase4-results.md) · [Phase 5](docs/phase5.md) · [Phase 6](docs/phase6.md) ·
[Limits suite](docs/limits.md), with every prediction graded against the results.

## Project phases

| phase | what I built | where |
|---|---|---|
| 1 | PyTorch basics: a character-level model; why parallel generation is fast | `scripts/`, [docs](docs/phase1.md) |
| 2 | A toy NCM from scratch: multiscale VQ tokenizer, coarse-to-fine generator, AR baseline | `src/ncm_lab/toy_ncm/`, [docs](docs/phase2.md) |
| 3 | Clean wrappers for the official NCM checkpoints, validated against the official evaluation | `src/ncm_lab/models/`, [docs](docs/phase3.md) |
| 4 | A fair benchmark: quality, latency, throughput and memory vs same-size AR models | `src/ncm_lab/bench/`, [docs](docs/phase4-results.md) |
| 5 | NCM as a tool: RAG compression, multilingual RAG, and a hand-written tool-calling agent | `src/ncm_lab/rag/`, `src/ncm_lab/agents/`, [docs](docs/phase5.md) |
| 6 | A production-style API: validation, timeouts, GPU lock, logging, metrics, Docker, CI | `src/ncm_lab/api/`, [docs](docs/phase6.md) |
| 7 | Limits suite: input/output length, batch size, faithfulness, robustness, sudoku, LLM, agent | `src/ncm_lab/limits/`, [docs](docs/limits.md) |
| – | A 12-page report generated from the result files | `paper/` |

## Tech stack

| area | tools |
|---|---|
| Modeling | PyTorch (CUDA, fp16 autocast), Hugging Face Transformers & Tokenizers, official NCM code (Hydra / OmegaConf) |
| Baselines | BART-base (XSum), opus-mt-de-en (MarianMT), qwen2.5:3b via Ollama |
| Evaluation | ROUGE (`rouge-score`), BLEU and chrF (`sacrebleu`), custom faithfulness and robustness checks |
| RAG & agents | sentence-transformers (MiniLM), FAISS, XQuAD, a plain-Python tool-calling loop |
| Serving | FastAPI, Pydantic, pydantic-settings, Uvicorn, Prometheus metrics, structured JSON logs |
| Engineering | pytest (55 CPU tests, GPU tests marked separately), ruff, pre-commit, GitHub Actions, Docker Compose with GPU |
| Analysis & writing | matplotlib, LaTeX |
| Hardware | RTX 3050 Laptop GPU (6 GB), WSL2 Ubuntu 24.04, CUDA 13 |

## Architecture

```mermaid
flowchart LR
    client([client]) -->|HTTP| api[FastAPI service<br/>src/ncm_lab/api]
    api --> sum[NCM summarizer<br/>XSum 88M + 61M]
    api --> tr[NCM translator<br/>WMT14 120M + 144M]
    api --> sud[NCM sudoku solver]
    api --> rag[RAG pipeline]
    api --> agent[agent loop]
    rag --> store[(FAISS + MiniLM<br/>XQuAD paragraphs)]
    rag --> llm[Ollama<br/>qwen2.5:3b]
    agent --> llm
    agent -->|tool calls| sum & tr & sud & store
    sum & tr & sud -.wraps.-> official[third_party/ncm<br/>official code]
```

## Quick start

**No GPU needed.** The API and the test suite run with tiny stand-in models:

```bash
git clone https://github.com/Aymenrahmanii/NCM.git ncm-lab && cd ncm-lab
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]"
NCM_BACKEND=fake pytest -q -m "not gpu"                  # 55 tests
NCM_BACKEND=fake uvicorn ncm_lab.api.main:app            # docs at http://localhost:8000/docs
```

**Full reproduction** (NVIDIA GPU, WSL2 Ubuntu):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130 && pip install -e ".[dev]"
bash scripts/download_data.sh && bash scripts/download_ncm.sh          # ~2.5 GB
bash scripts/install_ollama_user.sh && (bash scripts/ollama.sh serve &) && bash scripts/ollama.sh pull qwen2.5:3b
bash scripts/reproduce_results.sh phase2-train                          # toy NCM, ~1 h
bash scripts/reproduce_results.sh                                       # every number in results/
python -m ncm_lab.limits.run --list                                     # the 8 limits experiments
bash paper/build.sh                                                     # rebuild the report
```

**Serve the real models:**

```bash
cp .env.example .env
uvicorn ncm_lab.api.main:app --port 8000        # /summarize /translate /sudoku/solve /rag/query /agent/run
docker compose up --build                       # or with Docker (GPU via the NVIDIA Container Toolkit)
```

## Repository structure

```
src/ncm_lab/
  toy_ncm/    toy NCM from scratch (phase 2)
  models/     wrappers for the official checkpoints + AR baselines
  bench/      benchmark harness and charts
  rag/        retrieval, RAG pipeline, experiments
  agents/     tool-calling agent and its tools
  api/        FastAPI service
  limits/     the 8 limits experiments + report generator
tests/        CPU tests (fake models) and GPU tests
scripts/      setup, downloads, reproduction, profiling
results/      every CSV, chart and model output quoted in the docs
docs/         one write-up per phase
paper/        LaTeX report, generated tables, PDF
```

## Limitations

- **One GPU, one machine.** Every speed number is from an RTX 3050 6 GB laptop under WSL2.
  Results on an H100 or with long outputs can differ.
- **Pretrained baselines.** BART-base and opus-mt start from large-scale pretraining; the NCM
  checkpoints were trained from scratch on XSum / WMT14, so the quality gap mixes method and
  pretraining.
- **Small evaluation sets.** 100–500 examples per task, 40 RAG questions, six agent tasks:
  enough for large effects, not small ones.
- **Simple metrics.** RAG accuracy uses lenient string matching; faithfulness uses string
  proxies, not human judgment.
- **No diffusion baseline.** The 31× claim itself was not re-measured.
- **NCM outputs hallucinate** names and figures; do not use them unchecked.

## Acknowledgements

NCM code and checkpoints: [Hazy Research](https://github.com/HazyResearch/ncm). Datasets: XSum,
WMT14, XQuAD, TinyStories and Tiny Shakespeare. Parts of the code, documentation and report were
written with the help of an AI assistant (Claude, Anthropic); I ran every experiment, and every
number comes from the files in `results/`.

## License

[MIT](LICENSE) for this repository's code and documentation. The official NCM code
(`third_party/ncm`, downloaded separately), the model checkpoints and the datasets keep their own
licenses.
