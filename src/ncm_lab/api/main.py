"""FastAPI service around the NCM models, the RAG pipeline and the agent.

    uvicorn ncm_lab.api.main:app --host 0.0.0.0 --port 8000
    NCM_BACKEND=fake uvicorn ncm_lab.api.main:app     # no GPU needed

Interactive docs at http://localhost:8000/docs
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import requests
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from . import registry as reg
from .schemas import (AgentRunRequest, AgentRunResponse, ErrorResponse, HealthResponse,
                      RAGQueryRequest, RAGQueryResponse, SummarizeRequest, TextsResponse,
                      TranslateRequest)
from .settings import get_settings

log = logging.getLogger("ncm_lab.api")

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    LATENCY = Histogram("ncm_request_latency_seconds", "Request latency", ["endpoint"])
    REQUESTS = Counter("ncm_requests_total", "Requests", ["endpoint", "status"])
except ImportError:  # metrics are optional
    LATENCY = REQUESTS = None


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, with any `extra=` fields included."""

    skip = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}

    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": round(record.created, 3), "level": record.levelname,
                "logger": record.name, "msg": record.getMessage()}
        data.update({k: v for k, v in record.__dict__.items() if k not in self.skip})
        return json.dumps(data, default=str)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("ncm_lab")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    start = time.perf_counter()
    app.state.registry = reg.load(settings)  # load once at startup, not per request
    app.state.gpu_lock = asyncio.Lock()      # one model call at a time on the 6 GB GPU
    log.info("startup complete", extra={"backend": settings.backend,
                                        "load_s": round(time.perf_counter() - start, 1)})
    yield


app = FastAPI(title="NCM Lab API", version="0.1.0", lifespan=lifespan,
              description="Next-Concept Models as fast specialist tools: summarization, "
                          "translation, RAG and an agent.")


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = rid
    endpoint = request.url.path
    log.info("request", extra={"request_id": rid, "method": request.method, "path": endpoint,
                               "status": response.status_code, "ms": round(ms, 1)})
    if LATENCY is not None and endpoint != "/metrics":
        LATENCY.labels(endpoint).observe(ms / 1000)
        REQUESTS.labels(endpoint, str(response.status_code)).inc()
    return response


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    body = ErrorResponse(error=f"http_{exc.status_code}", detail=str(exc.detail),
                         request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def call_model(request: Request, fn, *args):
    """Run blocking model code in a worker thread with a timeout and clear errors."""
    settings = get_settings()
    async with request.app.state.gpu_lock:
        try:
            return await asyncio.wait_for(run_in_threadpool(fn, *args),
                                          timeout=settings.request_timeout_s)
        except asyncio.TimeoutError:
            # note: the worker thread finishes in the background; the client gets 504 now
            raise HTTPException(504, f"model call took longer than {settings.request_timeout_s}s")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "GPU out of memory; retry with fewer or shorter texts")
        except requests.RequestException as e:
            raise HTTPException(502, f"LLM backend unavailable: {type(e).__name__}")


def registry(request: Request) -> reg.Registry:
    return request.app.state.registry


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    r = registry(request)
    llm_ok = await run_in_threadpool(r.llm.available)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(status="ok", backend=r.backend, models_loaded=r.loaded, gpu=gpu,
                          llm_reachable=llm_ok)


@app.post("/summarize", response_model=TextsResponse,
          responses={503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
async def summarize(body: SummarizeRequest, request: Request) -> TextsResponse:
    start = time.perf_counter()
    outputs = await call_model(request, registry(request).summarizer.generate, body.texts)
    return TextsResponse(outputs=outputs, model="ncm-xsum",
                         latency_ms=round((time.perf_counter() - start) * 1000, 1))


@app.post("/translate", response_model=TextsResponse,
          responses={503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
async def translate(body: TranslateRequest, request: Request) -> TextsResponse:
    start = time.perf_counter()
    tools = registry(request).toolbox
    outputs = await call_model(request, lambda ts: [tools.translate_de_en(t) for t in ts],
                               body.texts)
    return TextsResponse(outputs=outputs, model="ncm-wmt14",
                         latency_ms=round((time.perf_counter() - start) * 1000, 1))


@app.post("/rag/query", response_model=RAGQueryResponse,
          responses={502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
async def rag_query(body: RAGQueryRequest, request: Request) -> RAGQueryResponse:
    pipe = registry(request).rag(body.k, body.compress)
    res = await call_model(request, pipe.query, body.question)
    return RAGQueryResponse(answer=res.answer, contexts=res.contexts,
                            timings_ms={k: round(v, 1) for k, v in res.timings_ms.items()})


@app.post("/agent/run", response_model=AgentRunResponse,
          responses={502: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
async def agent_run(body: AgentRunRequest, request: Request) -> AgentRunResponse:
    from ncm_lab.agents.loop import run_agent

    r = registry(request)
    out = await call_model(request, run_agent, body.task, r.llm, r.toolbox, body.max_steps)
    return AgentRunResponse(**{k: out[k] for k in ("answer", "steps", "total_ms", "llm_ms",
                                                    "ncm_ms")})


@app.get("/metrics", include_in_schema=False)
async def metrics():
    if LATENCY is None:
        raise HTTPException(404, "prometheus_client not installed")
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
