"""Phase 5 use case 3: run multi-step tasks through the agent and log every step's latency.

    python -m ncm_lab.agents.run_tasks
"""

import json
import pickle
from pathlib import Path

from ncm_lab.rag.data import eval_questions, is_correct, load_xquad, paragraphs
from ncm_lab.rag.llm import OllamaLLM

from .loop import run_agent
from .tools import Toolbox

RESULTS = Path("results")


def build_tasks() -> list[dict]:
    qs = eval_questions(40)
    de_par = paragraphs("de")[qs[5]["paragraph"]]
    de_q = {r["id"]: r["question"] for r in load_xquad("de")}
    puzzle_ids = pickle.load(open("data/sudoku/hard/puzzles.pickle", "rb"))[0]
    puzzle = "".join(str(t) for t in puzzle_ids if t < 10)
    return [
        {"name": "search_answer", "gold": qs[0]["answers"],
         "task": f"Use the documents to answer: {qs[0]['question']}"},
        {"name": "search_summarize_answer", "gold": qs[10]["answers"],
         "task": f"Search the documents for '{qs[10]['question']}', summarize the most relevant "
                 "passage in one sentence, then answer the question."},
        {"name": "translate_summarize", "gold": None,
         "task": "Translate this German text to English, then give me a one-sentence summary "
                 f"of it.\n\n{de_par}"},
        {"name": "german_question_rag", "gold": qs[20]["answers"],
         "task": "Translate this German question to English and answer it using the documents: "
                 f"{de_q[qs[20]['id']]}"},
        {"name": "sudoku", "gold": None,
         "task": f"Solve this sudoku (0 = empty cell, 81 digits row by row): {puzzle}. "
                 "Tell me the middle row of the solution and whether the solution is valid."},
        {"name": "two_questions", "gold": qs[30]["answers"] + qs[31]["answers"],
         "task": f"Answer both questions using the documents. 1) {qs[30]['question']} "
                 f"2) {qs[31]['question']}"},
    ]


def main() -> None:
    llm = OllamaLLM()
    if not llm.available():
        raise SystemExit("Ollama is not reachable; start it with: bash scripts/ollama.sh serve")
    tools = Toolbox()
    tools.warm_up()
    llm.ask("hi")  # load the LLM before timing

    RESULTS.mkdir(exist_ok=True)
    runs = []
    with open(RESULTS / "phase5_agent_log.jsonl", "w", encoding="utf-8") as f:
        for t in build_tasks():
            run = run_agent(t["task"], llm, tools) | {"name": t["name"]}
            run["correct"] = is_correct(run["answer"], t["gold"]) if t["gold"] else None
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
            runs.append(run)
            tools_used = [s["name"] for s in run["steps"] if s["kind"] == "tool"]
            print(f"{t['name']}: {run['total_ms']:.0f} ms, tools {tools_used}, "
                  f"correct={run['correct']}\n  answer: {run['answer'][:200]}", flush=True)

    total = sum(r["total_ms"] for r in runs)
    llm_ms, ncm_ms = sum(r["llm_ms"] for r in runs), sum(r["ncm_ms"] for r in runs)
    other = sum(r["other_tool_ms"] for r in runs)
    lines = ["# Phase 5: agent with NCM tools", "",
             "| task | steps | tools called | total ms | LLM ms | NCM tools ms | correct |",
             "|---|---|---|---|---|---|---|"]
    for r in runs:
        tools_used = ", ".join(s["name"] for s in r["steps"] if s["kind"] == "tool") or "-"
        lines.append(f"| {r['name']} | {len(r['steps'])} | {tools_used} | {r['total_ms']:.0f} "
                     f"| {r['llm_ms']:.0f} | {r['ncm_ms']:.0f} | {r['correct']} |")
    lines += ["", f"Share of total time: LLM {llm_ms / total:.1%}, NCM tools {ncm_ms / total:.1%}, "
                  f"search {other / total:.1%}, loop overhead "
                  f"{(total - llm_ms - ncm_ms - other) / total:.1%}."]
    lines += ["", "## Answers", ""] + [f"- **{r['name']}**: {r['answer']}" for r in runs]
    (RESULTS / "phase5_agent_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
