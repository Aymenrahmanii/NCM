"""A tool-calling agent loop in plain Python (no framework).

1. send the conversation + tool schemas to the LLM
2. if it asks for tools, run them and append the results as `tool` messages
3. repeat until the LLM answers without asking for a tool
"""

import json
import time

from ncm_lab.rag.llm import OllamaLLM

from .tools import NCM_TOOLS, SCHEMAS, Toolbox

SYSTEM = """You are an assistant that answers by calling tools. Rules:
- A question about facts or "the documents": call search_docs first, then answer only from
  its passages. Never answer facts from memory.
- Two questions: call search_docs once per question.
- German text or a German question: call translate_de_en first.
- A sudoku: call solve_sudoku with the 81 digits exactly as given. Never solve it yourself.
- Call summarize_fast only when the user asks for a summary.
When you have what you need, reply with a short final answer and no tool call."""


def run_agent(task: str, llm: OllamaLLM, tools: Toolbox, max_steps: int = 8) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    steps = []
    start = time.perf_counter()
    answer = "(stopped: too many steps)"
    for _ in range(max_steps):
        reply = llm.chat(messages, tools=SCHEMAS)
        steps.append({"kind": "llm", "ms": round(reply.ms, 1),
                      "prompt_tokens": reply.prompt_tokens,
                      "completion_tokens": reply.completion_tokens,
                      "requested": [c["function"]["name"] for c in reply.tool_calls]})
        messages.append(reply.raw_message)
        if not reply.tool_calls:
            answer = reply.content
            break
        for call in reply.tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name in tools.functions:
                result, ms = tools.call(name, args)
            else:
                result, ms = f"error: unknown tool {name}", 0.0
            steps.append({"kind": "tool", "name": name, "ncm": name in NCM_TOOLS,
                          "args": {k: str(v)[:200] for k, v in args.items()},
                          "ms": round(ms, 1), "result": result[:400]})
            messages.append({"role": "tool", "content": result, "tool_name": name})
    total = (time.perf_counter() - start) * 1000
    return {"task": task, "answer": answer, "steps": steps, "total_ms": round(total, 1),
            "llm_ms": round(sum(s["ms"] for s in steps if s["kind"] == "llm"), 1),
            "ncm_ms": round(sum(s["ms"] for s in steps if s.get("ncm")), 1),
            "other_tool_ms": round(sum(s["ms"] for s in steps
                                       if s["kind"] == "tool" and not s["ncm"]), 1)}
