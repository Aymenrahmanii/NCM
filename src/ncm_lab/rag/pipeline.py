"""Retrieve -> (optionally compress each chunk) -> answer with the LLM. Every stage is timed."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .llm import OllamaLLM
from .store import VectorStore

PROMPT = """Answer the question using only the context below. Reply with a short answer \
(a few words), no explanation.

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str]
    doc_ids: list[int]
    prompt_tokens: int
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, store: VectorStore, llm: OllamaLLM, k: int = 3,
                 compressor: Callable[[list[str]], list[str]] | None = None):
        self.store = store
        self.llm = llm
        self.k = k
        self.compressor = compressor

    def query(self, question: str) -> RAGResult:
        timings = {}
        t = time.perf_counter()
        hits = self.store.search(question, self.k)
        timings["retrieve"] = (time.perf_counter() - t) * 1000

        chunks = [h.text for h in hits]
        if self.compressor is not None:
            t = time.perf_counter()
            chunks = self.compressor(chunks)  # one short sentence per chunk
            timings["compress"] = (time.perf_counter() - t) * 1000

        context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
        reply = self.llm.ask(PROMPT.format(context=context, question=question))
        timings["llm"] = reply.ms
        timings["total"] = sum(timings.values())
        return RAGResult(question, reply.content, chunks, [h.doc_id for h in hits],
                         reply.prompt_tokens, timings)
