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


def test_metrics(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ncm_request_latency_seconds" in r.text
