# Phase 6: production layer

> Draft written with Claude. Rewrite it in your own words.

The model stays experimental; the service around it is built the way teams build services.

## What is where

| concern | how | file |
|---|---|---|
| API | FastAPI: `/summarize`, `/translate`, `/sudoku/solve`, `/rag/query`, `/agent/run`, `/health`, `/metrics`; docs at `/docs` | `src/ncm_lab/api/main.py` |
| input validation | Pydantic schemas: empty / blank / too many / too long texts → 422 with a clear message | `api/schemas.py` |
| configuration | pydantic-settings, `NCM_*` environment variables or `.env`; `.env.example` committed, `.env` ignored | `api/settings.py` |
| model lifecycle | all models load once at startup (FastAPI lifespan), never per request | `api/registry.py` |
| GPU safety | one model call at a time (asyncio lock, held until the worker thread really ends, even after a timeout); timeout → 504; GPU still busy → 503; CUDA out-of-memory → 503 with advice, cache emptied; LLM down → 502; any other error → 500 in the same JSON shape, logged with its `request_id` | `api/main.py` |
| logging | one JSON object per line with `request_id`, path, status, latency; the id is echoed in `x-request-id` | `api/main.py` |
| metrics | Prometheus histogram of latency per endpoint + request counter at `/metrics` | `api/main.py` |
| tests without a GPU | `NCM_BACKEND=fake` swaps in tiny stand-in models with the same interface | `api/registry.py` |
| tests | 54 CPU tests (API, validation, error paths, agent loop with a scripted LLM, RAG helpers, toy NCM, every limits experiment with stand-in models) + 3 GPU tests (wrapper, ROUGE regression) | `tests/` |
| quality regression | fails if ROUGE-L on 20 fixed XSum articles drops below 22 (measured: 27.1) | `tests/test_quality_regression.py` |
| code quality | ruff (lint) via pre-commit | `pyproject.toml`, `.pre-commit-config.yaml` |
| container | non-root user, health check, weights mounted read-only, small build context | `Dockerfile`, `.dockerignore` |
| orchestration | `docker compose up` starts the API and Ollama with GPU access | `docker-compose.yml` |
| CI | GitHub Actions: CPU PyTorch, ruff, `pytest -m "not gpu"` on every push | `.github/workflows/ci.yml` |

## Verified on this machine

- All endpoints on the real models (local uvicorn): `/health` lists the 5 loaded components and
  the GPU; `/summarize`, `/translate`, `/rag/query`, `/agent/run` answer; bad input → 422;
  `/metrics` counts every endpoint.
- The Docker image builds and runs: healthy health check, non-root user, `/docs` served
  (fake backend, because the GPU is not yet available to Docker here, see below).
- 23 CPU tests and 3 GPU tests pass; ruff passes.

## Lessons

- **`.dockerignore` matters.** The first build sent the whole folder (6 GB virtualenv,
  2.5 GB of weights) to Docker as build context; with `.dockerignore` it is 106 KB.
- **Docker Engine inside WSL needs the NVIDIA Container Toolkit** for `gpus:` in compose
  (`could not select device driver "nvidia"`). Install it once with
  `bash scripts/setup_docker_gpu.sh` (needs sudo). Docker Desktop has it built in.
- **A timeout does not stop a GPU call.** `asyncio.wait_for` returns 504 to the client but
  the worker thread keeps running. The GPU lock is therefore released only when the thread
  ends (not when the client gets its 504), so a timed-out call can never overlap the next one;
  a real deployment would run models in a separate worker process that can be killed.
- **Cold start is slow:** loading three NCM models and the vector store takes ~1 minute,
  and the first LLM call takes ~8 s while Ollama loads the model. The Docker health check
  allows 3 minutes before it starts counting failures.
- **XSum invents details.** Asked to summarize an article about a £20m bridge over the
  river Taff, it answered "Plans for a £12m bridge in Cardiff have been approved." An API
  serving it should say so.
