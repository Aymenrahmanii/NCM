"""Limit 7: a 150-260M specialist (NCM) vs a 3B general LLM (qwen2.5:3b via Ollama).

Same examples, one request at a time, text in -> text out (tokenization and, for the LLM,
the local HTTP call included, so these latencies are end-to-end and a little higher than
Phase 4's generation-only timings). The AR baseline is included as the third point. The LLM is
zero-shot with a one-line instruction and is 4-bit quantized by Ollama; the specialists were
trained on these exact tasks.
"""

import re

from .common import (free_gpu, load_model, load_task, model_label, pct, quality, timed, words,
                     write)

PROMPTS = {
    "xsum": "Summarize this BBC news article in one sentence of about 20 words. Reply with "
            "the sentence only.\n\n{text}",
    "wmt14": "Translate this German text into English. Reply with the translation only.\n\n"
             "{text}",
}


def clean_llm(text: str) -> str:
    """First non-empty line, without a 'Summary:'-style label or wrapping quotes."""
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^(summary|translation|english)\s*:\s*", "", line.strip(), flags=re.I)
    return line.strip().strip('"“”')


def run_one_by_one(generate_one, inputs: list) -> tuple[list[str], list[float]]:
    outs, ms = [], []
    for x in inputs:
        out, t = timed(generate_one, x)
        outs.append(out)
        ms.append(t)
    return outs, ms


def row(task: str, system: str, params_M, hyps: list[str], refs: list[str],
        ms: list[float]) -> dict:
    return {"task": task, "system": system, "params_M": params_M, "n": len(hyps),
            **quality(task, hyps, refs), "output_words": words(hyps),
            "latency_p50_ms": pct(ms, 0.5), "latency_p95_ms": pct(ms, 0.95)}


def llm_generate(llm, task: str):
    return lambda text: clean_llm(llm.ask(PROMPTS[task].format(text=text)).content)


def main(n: int = 100, precision: str = "fp16", llm_params_M: float = 3090) -> None:
    from ncm_lab.rag.llm import OllamaLLM

    llm = OllamaLLM()
    if not llm.available():
        raise SystemExit("Ollama is not reachable; start it with: bash scripts/ollama.sh serve")
    rows, details = [], []
    for task in ("xsum", "wmt14"):
        ncm_inputs, texts, refs = load_task(task, n)
        systems = []
        for kind in ("ncm", "ar"):
            model = load_model(kind, task, precision)
            model.generate([texts[0]] if kind == "ar" else [ncm_inputs[0]])  # warm-up
            hyps, ms = run_one_by_one(lambda x, m=model: m.generate([x])[0],
                                      ncm_inputs if kind == "ncm" else texts)
            params = getattr(model, "n_params_total", None) or model.n_params
            systems.append((model_label(kind, task), round(params / 1e6), hyps, ms))
            del model
            free_gpu()
        llm.ask("hi")  # load the LLM before timing
        hyps, ms = run_one_by_one(llm_generate(llm, task), texts)
        systems.append((f"{llm.model} (zero-shot)", llm_params_M, hyps, ms))
        for name, params, hyps, ms in systems:
            rows.append(row(task, name, params, hyps, refs, ms))
            details += [{"task": task, "system": name, "hyp": h, "ref": r}
                        for h, r in list(zip(hyps, refs))[:10]]
    write("llm_vs_ncm", rows, details)
