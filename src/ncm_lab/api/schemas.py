"""Request and response schemas: bad input is rejected with a clear 422 error."""

from pydantic import BaseModel, Field, field_validator

MAX_TEXTS = 32
MAX_CHARS = 20_000


class TextsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS,
                             examples=[["Die Katze schläft auf dem Sofa."]])

    @field_validator("texts")
    @classmethod
    def non_empty(cls, texts: list[str]) -> list[str]:
        for i, t in enumerate(texts):
            if not t.strip():
                raise ValueError(f"texts[{i}] is empty")
            if len(t) > MAX_CHARS:
                raise ValueError(f"texts[{i}] is longer than {MAX_CHARS} characters")
        return texts


class SummarizeRequest(TextsRequest):
    pass


class TranslateRequest(TextsRequest):
    pass


class SudokuRequest(BaseModel):
    puzzles: list[str] = Field(..., min_length=1, max_length=MAX_TEXTS, examples=[[
        "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]])

    @field_validator("puzzles")
    @classmethod
    def eighty_one_cells(cls, puzzles: list[str]) -> list[str]:
        out = []
        for i, p in enumerate(puzzles):
            cells = "".join(p.split())  # line breaks and spaces between rows are fine
            if len(cells) != 81 or any(c not in "0123456789." for c in cells):
                raise ValueError(f"puzzles[{i}] must be 81 cells of 1-9, with 0 or . for blanks")
            out.append(cells)
        return out


class SudokuSolution(BaseModel):
    solution: str
    valid: bool


class SudokuResponse(BaseModel):
    solutions: list[SudokuSolution]
    model: str
    latency_ms: float


class TextsResponse(BaseModel):
    outputs: list[str]
    model: str
    latency_ms: float


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          examples=["What is the main language of Victoria?"])
    k: int = Field(3, ge=1, le=10)
    compress: bool = Field(False, description="compress each retrieved chunk with NCM first")


class RAGQueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    timings_ms: dict[str, float]


class AgentRunRequest(BaseModel):
    task: str = Field(..., min_length=3, max_length=5000)
    max_steps: int = Field(6, ge=1, le=10)


class AgentStep(BaseModel):
    kind: str
    ms: float
    name: str | None = None
    ncm: bool | None = None
    args: dict | None = None
    result: str | None = None
    requested: list[str] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class AgentRunResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    total_ms: float
    llm_ms: float
    ncm_ms: float


class HealthResponse(BaseModel):
    status: str
    backend: str
    models_loaded: list[str]
    gpu: str | None
    llm_reachable: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
