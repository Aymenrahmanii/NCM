"""Tools the agent can call. Three of them are NCM models used as fast specialists."""

import time

from ncm_lab.rag.data import paragraphs, split_sentences

NCM_TOOLS = {"summarize_fast", "translate_de_en", "solve_sudoku"}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the English Wikipedia document collection. Returns the most "
                       "relevant passages.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "summarize_fast",
        "description": "Summarize an English text into one sentence.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the text to summarize"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "translate_de_en",
        "description": "Translate German text into English.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "German text"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "solve_sudoku",
        "description": "Solve a 9x9 sudoku. Give the puzzle as 81 digits, row by row, with 0 "
                       "for empty cells.",
        "parameters": {"type": "object", "properties": {
            "puzzle": {"type": "string", "description": "81 digits, 0 = empty"}},
            "required": ["puzzle"]}}},
]


class Toolbox:
    """Loads every model once; `call` runs a tool and returns (result text, milliseconds)."""

    def __init__(self, precision: str = "fp16", k: int = 3, store=None, summarizer=None,
                 translator=None, sudoku=None):
        """Pass already-loaded models (the API does) or let the toolbox load its own."""
        from ncm_lab.models.ncm import NCMSudokuSolver, NCMSummarizer, NCMTranslator
        from ncm_lab.rag.store import VectorStore

        self.k = k
        if store is None:
            store = VectorStore()
            store.add(paragraphs("en"))
        self.store = store
        self.summarizer = summarizer or NCMSummarizer(precision=precision, batch_size=8)
        self.translator = translator or NCMTranslator(precision=precision, batch_size=32)
        self.sudoku = sudoku or NCMSudokuSolver("hard", precision=precision)
        self.functions = {"search_docs": self.search_docs, "summarize_fast": self.summarize_fast,
                          "translate_de_en": self.translate_de_en,
                          "solve_sudoku": self.solve_sudoku}

    def warm_up(self) -> None:
        self.search_docs("test")
        self.summarize_fast("The council met on Monday. It approved the new budget.")
        self.translate_de_en("Guten Morgen.")
        self.solve_sudoku("0" * 81)

    def search_docs(self, query: str) -> str:
        hits = self.store.search(query, self.k)
        return "\n\n".join(f"[doc {h.doc_id}] {h.text}" for h in hits)

    def summarize_fast(self, text: str) -> str:
        return self.summarizer.generate([text])[0]

    def translate_de_en(self, text: str) -> str:
        # NCM WMT14 reads at most 96 tokens, so translate sentence by sentence
        return " ".join(self.translator.generate(split_sentences(text) or [text]))

    def solve_sudoku(self, puzzle: str) -> str:
        solution = self.sudoku.generate([puzzle])[0]
        valid = self.sudoku.is_valid(solution, puzzle)
        rows = "\n".join(solution[9 * i : 9 * i + 9] for i in range(9))
        return f"solution (row by row):\n{rows}\nvalid: {valid}"

    def call(self, name: str, args: dict) -> tuple[str, float]:
        start = time.perf_counter()
        try:
            result = self.functions[name](**args)
        except Exception as e:  # report tool errors back to the LLM instead of crashing
            result = f"error: {type(e).__name__}: {e}"
        return result, (time.perf_counter() - start) * 1000
