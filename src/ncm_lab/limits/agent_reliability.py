"""Limit 8: how reliably does the LLM drive the NCM tools? Many sampled runs per task.

Phase 5 ran each agent task once at temperature 0. Here every task runs `repeats` times at
temperature 0.7 with different seeds (plus once at temperature 0), and we count how often the
agent calls the tools the task needs, sends arguments a tool rejects, runs out of steps, and
(where there is a gold answer) gets it right.
"""

from collections import Counter

from .common import mean, write

# tools a correct run must call (at least this many times)
EXPECTED_TOOLS = {
    "search_answer": {"search_docs": 1},
    "search_summarize_answer": {"search_docs": 1, "summarize_fast": 1},
    "translate_summarize": {"translate_de_en": 1, "summarize_fast": 1},
    "german_question_rag": {"translate_de_en": 1, "search_docs": 1},
    "sudoku": {"solve_sudoku": 1},
    "two_questions": {"search_docs": 2},
}


def analyse(run: dict, task: dict) -> dict:
    from ncm_lab.rag.data import is_correct

    tool_steps = [s for s in run["steps"] if s["kind"] == "tool"]
    used = Counter(s["name"] for s in tool_steps)
    expected = EXPECTED_TOOLS.get(task["name"], {})
    return {"right_tools": all(used[t] >= k for t, k in expected.items()),
            "tool_errors": sum(s["result"].startswith("error") for s in tool_steps),
            "out_of_steps": run["out_of_steps"],
            "correct": is_correct(run["answer"], task["gold"]) if task.get("gold") else None,
            "llm_calls": sum(s["kind"] == "llm" for s in run["steps"]),
            "total_ms": run["total_ms"]}


def summarize(task_name: str, temperature: float, results: list[dict]) -> dict:
    graded = [r["correct"] for r in results if r["correct"] is not None]
    return {"task": task_name, "temperature": temperature, "runs": len(results),
            "right_tools": mean(r["right_tools"] for r in results),
            "runs_with_tool_error": mean(r["tool_errors"] > 0 for r in results),
            "out_of_steps": mean(r["out_of_steps"] for r in results),
            "accuracy": mean(graded) if graded else None,
            "mean_llm_calls": mean(r["llm_calls"] for r in results),
            "mean_total_ms": round(mean(r["total_ms"] for r in results), 1)}


def evaluate(tasks: list[dict], make_llm, tools, repeats: int = 10, temperature: float = 0.7,
             max_steps: int = 6) -> tuple[list[dict], list[dict]]:
    """make_llm(temperature, seed) -> LLM client."""
    from ncm_lab.agents.loop import run_agent

    rows, details = [], []
    for task in tasks:
        for temp, seeds in [(0.0, [0]), (temperature, range(repeats))]:
            results = []
            for seed in seeds:
                run = run_agent(task["task"], make_llm(temp, seed), tools, max_steps)
                a = analyse(run, task)
                results.append(a)
                details.append({"task": task["name"], "temperature": temp, "seed": seed,
                                "answer": run["answer"],
                                "tools": [s["name"] for s in run["steps"] if s["kind"] == "tool"]}
                               | a)
            rows.append(summarize(task["name"], temp, results))
    return rows, details


def main(n: int = 10, precision: str = "fp16") -> None:
    from ncm_lab.agents.run_tasks import build_tasks
    from ncm_lab.agents.tools import Toolbox
    from ncm_lab.rag.llm import OllamaLLM

    if not OllamaLLM().available():
        raise SystemExit("Ollama is not reachable; start it with: bash scripts/ollama.sh serve")
    tools = Toolbox(precision=precision)
    tools.warm_up()
    OllamaLLM().ask("hi")
    rows, details = evaluate(build_tasks(), lambda t, s: OllamaLLM(temperature=t, seed=s),
                             tools, repeats=n)
    write("agent_reliability", rows, details)
