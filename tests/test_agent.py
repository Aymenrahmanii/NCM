"""Agent loop tests with a scripted LLM: no GPU, no Ollama."""

from ncm_lab.agents.loop import parse_args, run_agent
from ncm_lab.agents.tools import Toolbox
from ncm_lab.api.registry import FakeStore, FakeSudoku, FakeTextModel
from ncm_lab.rag.llm import LLMReply


def toolbox() -> Toolbox:
    return Toolbox(store=FakeStore(), summarizer=FakeTextModel("summary"),
                   translator=FakeTextModel("translation"), sudoku=FakeSudoku("x"))


def call(name: str, arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


class ScriptedLLM:
    """Replies with the scripted tool calls in order, then with a final answer."""

    def __init__(self, *turns: list[dict]):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools=None, temperature=0.0):
        self.seen.append({"messages": list(messages), "tools": tools})
        if tools is not None and self.turns:
            return LLMReply("", tool_calls=self.turns.pop(0), ms=1.0,
                            raw_message={"role": "assistant", "content": ""})
        return LLMReply("final answer", ms=1.0,
                        raw_message={"role": "assistant", "content": "final answer"})


def tool_results(llm: ScriptedLLM) -> list[str]:
    return [m["content"] for m in llm.seen[-1]["messages"] if m["role"] == "tool"]


def test_parse_args():
    assert parse_args({"query": "x"}) == {"query": "x"}
    assert parse_args('{"query": "x"}') == {"query": "x"}
    assert parse_args("") == {} and parse_args(None) == {}
    assert parse_args("{not json") is None
    assert parse_args("[1, 2]") is None


def test_tool_results_reach_the_llm():
    llm = ScriptedLLM([call("search_docs", {"query": "Rhine Germany"})])
    out = run_agent("Where does the Rhine flow?", llm, toolbox())
    assert out["answer"] == "final answer"
    assert [s["kind"] for s in out["steps"]] == ["llm", "tool", "llm"]
    assert "Rhine flows through Germany" in tool_results(llm)[0]


def test_malformed_json_arguments_are_reported_not_raised():
    llm = ScriptedLLM([call("search_docs", "{bad json")])
    out = run_agent("q", llm, toolbox())
    assert out["answer"] == "final answer"
    assert tool_results(llm)[0].startswith("error: arguments for search_docs")


def test_unknown_tool_is_reported():
    llm = ScriptedLLM([call("browse_web", {"url": "x"})])
    run_agent("q", llm, toolbox())
    assert tool_results(llm) == ["error: unknown tool browse_web"]


def test_out_of_steps_still_answers_from_tool_results():
    looping = [[call("search_docs", {"query": "x"})] for _ in range(5)]
    llm = ScriptedLLM(*looping)
    out = run_agent("q", llm, toolbox(), max_steps=2)
    assert out["answer"] == "final answer"
    assert llm.seen[-1]["tools"] is None  # the last call cannot ask for more tools
    assert len(tool_results(llm)) == 2
    assert sum(s["kind"] == "llm" for s in out["steps"]) == 3


def test_sudoku_given_as_a_json_number_keeps_its_leading_zeros():
    seen = []

    class RecordingSudoku(FakeSudoku):
        def generate(self, puzzles):
            seen.extend(puzzles)
            return super().generate(puzzles)

    tools = toolbox()
    tools.sudoku = RecordingSudoku("x")
    puzzle = "0" * 9 + "1" * 72
    tools.call("solve_sudoku", {"puzzle": int(puzzle)})
    assert seen == [puzzle]


def test_translate_many_batches_sentences_and_keeps_text_boundaries():
    calls = []

    class RecordingTranslator(FakeTextModel):
        def generate(self, texts):
            calls.append(list(texts))
            return [f"<{t}>" for t in texts]

    tools = toolbox()
    tools.translator = RecordingTranslator("t")
    out = tools.translate_many(["Guten Morgen. Wie geht es?", "Am 3. Oktober war Feiertag."])
    assert calls == [["Guten Morgen.", "Wie geht es?", "Am 3. Oktober war Feiertag."]]
    assert out == ["<Guten Morgen.> <Wie geht es?>", "<Am 3. Oktober war Feiertag.>"]
