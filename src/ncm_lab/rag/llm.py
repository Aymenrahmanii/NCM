"""Minimal Ollama chat client (no framework), with token counts and timing."""

import os
import time
from dataclasses import dataclass, field

import requests

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class LLMReply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ms: float = 0.0
    raw_message: dict = field(default_factory=dict)


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 num_ctx: int = 4096, timeout: float = 120.0, temperature: float = 0.0,
                 seed: int = 0):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.temperature = temperature  # default for every call; 0 = deterministic
        self.seed = seed

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float | None = None) -> LLMReply:
        temperature = self.temperature if temperature is None else temperature
        body = {"model": self.model, "messages": messages, "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                            "seed": self.seed}}
        if tools:
            body["tools"] = tools
        start = time.perf_counter()
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        ms = (time.perf_counter() - start) * 1000
        data = r.json()
        msg = data.get("message", {})
        return LLMReply(content=msg.get("content", "").strip(),
                        tool_calls=msg.get("tool_calls") or [],
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0), ms=ms, raw_message=msg)

    def ask(self, prompt: str) -> LLMReply:
        return self.chat([{"role": "user", "content": prompt}])

    def available(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/version", timeout=3).ok
        except requests.RequestException:
            return False
