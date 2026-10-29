"""API tests against the fake backend: no GPU, no model downloads."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("NCM_BACKEND", "fake")
    from ncm_lab.api.settings import get_settings

    get_settings.cache_clear()
    from ncm_lab.api.main import app

    with TestClient(app) as c:
        yield c
    mp.undo()
    get_settings.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["backend"] == "fake"
    assert "x-request-id" in r.headers


def test_request_id_is_echoed(client):
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


def test_summarize_returns_one_output_per_text(client):
    r = client.post("/summarize", json={"texts": ["First article text.", "Second one."]})
    assert r.status_code == 200
    assert len(r.json()["outputs"]) == 2


@pytest.mark.parametrize("payload", [
    {},                                  # missing field
    {"texts": []},                       # empty list
    {"texts": ["   "]},                  # blank text
    {"texts": ["x"] * 33},               # too many texts
    {"texts": "not a list"},             # wrong type
])
def test_bad_input_is_rejected_with_422(client, payload):
    r = client.post("/summarize", json=payload)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_translate(client):
    r = client.post("/translate", json={"texts": ["Guten Morgen. Wie geht es dir?"]})
    assert r.status_code == 200
    assert r.json()["model"] == "ncm-wmt14"


def test_rag_query(client):
    r = client.post("/rag/query", json={"question": "What language is spoken in Victoria?",
                                        "k": 2, "compress": True})
    assert r.status_code == 200
    body = r.json()
    assert len(body["contexts"]) == 2
    assert {"retrieve", "compress", "llm", "total"} <= set(body["timings_ms"])


def test_rag_query_validates_k(client):
    assert client.post("/rag/query", json={"question": "Hello?", "k": 0}).status_code == 422


def test_agent_run(client):
    r = client.post("/agent/run", json={"task": "Say hello."})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["steps"][0]["kind"] == "llm"


SOLVED = "123456789456789123789123456214365897365897214897214365531642978642978531978531642"


def test_sudoku_solve(client):
    puzzle = "0" * 9 + SOLVED[9:]
    rows = "\n".join(puzzle[i : i + 9] for i in range(0, 81, 9))  # line breaks are allowed
    r = client.post("/sudoku/solve", json={"puzzles": [rows, "9" + SOLVED[1:]]})
    assert r.status_code == 200
    sols = r.json()["solutions"]
    assert sols[0] == {"solution": SOLVED, "valid": True}
    assert sols[1]["valid"] is False  # the fake ignores the clue "9", so it is not kept


@pytest.mark.parametrize("puzzle", ["123", "x" * 81, "1" * 82])
def test_sudoku_rejects_malformed_puzzles(client, puzzle):
    r = client.post("/sudoku/solve", json={"puzzles": [puzzle]})
    assert r.status_code == 422


def test_unexpected_error_is_json_with_request_id(client, monkeypatch):
    monkeypatch.setattr(client.app.state.registry.summarizer, "generate",
                        lambda texts: 1 / 0)
    r = client.post("/summarize", json={"texts": ["x"]}, headers={"x-request-id": "boom1"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "detail": "ZeroDivisionError",
                        "request_id": "boom1"}
    assert r.headers["x-request-id"] == "boom1"


def test_timed_out_call_keeps_the_gpu_lock_until_it_finishes(client, monkeypatch):
    import threading

    from ncm_lab.api import main
    from ncm_lab.api.settings import Settings

    monkeypatch.setattr(main, "get_settings",
                        lambda: Settings(backend="fake", request_timeout_s=0.2))
    summarizer = client.app.state.registry.summarizer
    first_running = threading.Event()
    release = threading.Event()

    def slow(texts):
        first_running.set()
        release.wait(5)
        first_running.clear()
        return ["slow"]

    monkeypatch.setattr(summarizer, "generate", slow)
    assert client.post("/summarize", json={"texts": ["x"]}).status_code == 504
    assert first_running.is_set()  # the timed-out call is still on the "GPU"

    # the next call must wait for the lock (and give up: busy) instead of running alongside
    monkeypatch.setattr(summarizer, "generate",
                        lambda texts: ["overlap" if first_running.is_set() else "alone"])
    r = client.post("/summarize", json={"texts": ["x"]})
    assert r.status_code == 503 and "busy" in r.json()["detail"]
    release.set()
    r = client.post("/summarize", json={"texts": ["x"]})
    assert r.status_code == 200 and r.json()["outputs"] == ["alone"]


def test_metrics(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ncm_request_latency_seconds" in r.text
