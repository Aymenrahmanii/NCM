"""A tool-calling agent loop in plain Python (no framework).

1. send the conversation + tool schemas to the LLM
2. if it asks for tools, run them and append the results as `tool` messages
3. repeat until the LLM answers without asking for a tool (or, out of steps, ask it to
   answer from what it has)
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

OUT_OF_STEPS = ("You have used all your tool calls. Give your final answer now from the tool "
                "results above, without calling a tool.")


def parse_args(raw) -> dict | None:
    """Tool arguments arrive as a dict or a JSON string; None if they are not a JSON object."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return None
    raw = {} if raw is None else raw
    return raw if isinstance(raw, dict) else None


def llm_step(reply) -> dict:
    return {"kind": "llm", "ms": round(reply.ms, 1), "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "requested": [c["function"]["name"] for c in reply.tool_calls]}


def run_agent(task: str, llm: OllamaLLM, tools: Toolbox, max_steps: int = 8) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    steps = []
    start = time.perf_counter()
    for _ in range(max_steps):
        reply = llm.chat(messages, tools=SCHEMAS)
        steps.append(llm_step(reply))
        messages.append(reply.raw_message)
        if not reply.tool_calls:
            answer = reply.content
            break
        for call in reply.tool_calls:
            name = call["function"]["name"]
            args = parse_args(call["function"].get("arguments"))
            if args is None:
                result, ms, args = f"error: arguments for {name} must be a JSON object", 0.0, {}
            elif name in tools.functions:
                result, ms = tools.call(name, args)
            else:
                result, ms = f"error: unknown tool {name}", 0.0
            steps.append({"kind": "tool", "name": name, "ncm": name in NCM_TOOLS,
                          "args": {k: str(v)[:200] for k, v in args.items()},
                          "ms": round(ms, 1), "result": result[:400]})
            messages.append({"role": "tool", "content": result, "tool_name": name})
    else:
        # out of steps: one last call without tools, so the tool results are not wasted
        reply = llm.chat(messages + [{"role": "user", "content": OUT_OF_STEPS}])
        steps.append(llm_step(reply))
        answer = reply.content or "(stopped: too many steps)"
    total = (time.perf_counter() - start) * 1000
    return {"task": task, "answer": answer, "steps": steps, "total_ms": round(total, 1),
            "llm_ms": round(sum(s["ms"] for s in steps if s["kind"] == "llm"), 1),
            "ncm_ms": round(sum(s["ms"] for s in steps if s.get("ncm")), 1),
            "other_tool_ms": round(sum(s["ms"] for s in steps
                                       if s["kind"] == "tool" and not s["ncm"]), 1)}
