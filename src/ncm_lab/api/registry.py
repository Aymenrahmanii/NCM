"""Everything the service needs, loaded once at startup (never per request)."""

import logging
from dataclasses import dataclass, field

from .settings import Settings

log = logging.getLogger("ncm_lab.api")


@dataclass
class Registry:
    backend: str
    summarizer: object
    translator: object
    sudoku: object
    store: object
    llm: object
    toolbox: object
    loaded: list[str] = field(default_factory=list)

    def rag(self, k: int, compress: bool):
        from ncm_lab.rag.pipeline import RAGPipeline

        return RAGPipeline(self.store, self.llm, k=k,
                           compressor=self.summarizer.generate if compress else None)


def load_real(settings: Settings) -> Registry:
    import os

    os.environ.setdefault("NCM_ARTIFACTS_DIR", settings.artifacts_dir)
    from ncm_lab.agents.tools import Toolbox
    from ncm_lab.models.ncm import NCMSudokuSolver, NCMSummarizer, NCMTranslator
    from ncm_lab.rag.data import paragraphs
    from ncm_lab.rag.llm import OllamaLLM
    from ncm_lab.rag.store import VectorStore

    kw = dict(device=settings.device, precision=settings.precision)
    log.info("loading models", extra={"precision": settings.precision})
    summarizer = NCMSummarizer(batch_size=8, **kw)
    translator = NCMTranslator(batch_size=32, **kw)
    sudoku = NCMSudokuSolver("hard", **kw)
    store = VectorStore()
    store.add(paragraphs("en"))
    llm = OllamaLLM(model=settings.ollama_model, host=settings.ollama_host,
                    timeout=settings.request_timeout_s)
    toolbox = Toolbox(k=settings.rag_top_k, store=store, summarizer=summarizer,
                      translator=translator, sudoku=sudoku)
    return Registry("real", summarizer, translator, sudoku, store, llm, toolbox,
                    ["ncm-xsum", "ncm-wmt14", "ncm-sudoku-hard", "vector-store", "llm-client"])


# ---- fakes: same interface, no GPU, no downloads (tests and CI) ---------------------------
class FakeTextModel:
    def __init__(self, task: str):
        self.task = task

    def generate(self, texts: list) -> list[str]:
        return [f"[{self.task}] " + " ".join(str(t).split()[:8]) for t in texts]


class FakeSudoku(FakeTextModel):
    def generate(self, puzzles: list) -> list[str]:
        return ["123456789456789123789123456214365897365897214897214365531642978642978531978531642"
                for _ in puzzles]

    @staticmethod
    def is_valid(solution: str, puzzle: str | None = None) -> bool:
        from ncm_lab.models.ncm import NCMSudokuSolver

        return NCMSudokuSolver.is_valid(solution, puzzle)


class FakeStore:
    docs = ["Victoria's main language is English.", "The Rhine flows through Germany.",
            "Sudoku is a logic puzzle with a 9x9 grid."]

    def search(self, query: str, k: int = 3):
        from ncm_lab.rag.store import Hit

        words = set(query.lower().split())
        ranked = sorted(range(len(self.docs)),
                        key=lambda i: -len(words & set(self.docs[i].lower().split())))
        return [Hit(self.docs[i], i, 1.0) for i in ranked[:k]]


class FakeLLM:
    def chat(self, messages, tools=None, temperature=0.0):
        from ncm_lab.rag.llm import LLMReply

        last = messages[-1]["content"]
        return LLMReply(content=f"fake answer to: {last[-60:]}", prompt_tokens=len(last.split()),
                        completion_tokens=5, ms=1.0,
                        raw_message={"role": "assistant", "content": "fake answer"})

    def ask(self, prompt: str):
        return self.chat([{"role": "user", "content": prompt}])

    def available(self) -> bool:
        return True


def load_fake(settings: Settings) -> Registry:
    from ncm_lab.agents.tools import Toolbox

    summarizer, translator = FakeTextModel("summary"), FakeTextModel("translation")
    sudoku, store = FakeSudoku("sudoku"), FakeStore()
    toolbox = Toolbox(store=store, summarizer=summarizer, translator=translator, sudoku=sudoku)
    return Registry("fake", summarizer, translator, sudoku, store, FakeLLM(), toolbox,
                    ["fake-summarizer", "fake-translator", "fake-sudoku", "fake-store",
                     "fake-llm"])


def load(settings: Settings) -> Registry:
    return load_fake(settings) if settings.backend == "fake" else load_real(settings)
