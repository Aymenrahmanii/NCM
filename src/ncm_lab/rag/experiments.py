"""Phase 5 RAG experiments (needs Ollama running with the LLM pulled).

    python -m ncm_lab.rag.experiments compression    # use case 1: NCM context compression
    python -m ncm_lab.rag.experiments multilingual   # use case 2: German docs via NCM translation
"""

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from .data import eval_questions, is_correct, paragraphs, split_sentences
from .llm import OllamaLLM
from .pipeline import RAGPipeline
from .store import VectorStore

RESULTS = Path("results")


def free_gpu() -> None:
    """Call after `del model` so the GPU memory goes back for the next model."""
    gc.collect()
    torch.cuda.empty_cache()


def run_questions(pipe: RAGPipeline, questions: list[dict], variant: str) -> list[dict]:
    pipe.query(questions[0]["question"])  # warm-up (loads the LLM, CUDA kernels)
    rows = []
    for q in questions:
        res = pipe.query(q["question"])
        rows.append({
            "variant": variant, "id": q["id"], "question": q["question"],
            "gold": q["answers"], "answer": res.answer,
            "correct": is_correct(res.answer, q["answers"]),
            "retrieved_gold": q["paragraph"] in res.doc_ids,
            "context_words": sum(len(c.split()) for c in res.contexts),
            "prompt_tokens": res.prompt_tokens,
            **{f"{k}_ms": round(v, 1) for k, v in res.timings_ms.items()},
            "contexts": res.contexts,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    mean = lambda k: round(statistics.fmean(r.get(k, 0.0) for r in rows), 1)  # noqa: E731
    return {
        "variant": rows[0]["variant"], "questions": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "retrieval_hit": round(sum(r["retrieved_gold"] for r in rows) / len(rows), 3),
        "context_words": mean("context_words"), "prompt_tokens": mean("prompt_tokens"),
        "retrieve_ms": mean("retrieve_ms"), "compress_ms": mean("compress_ms"),
        "llm_ms": mean("llm_ms"), "total_ms": mean("total_ms"),
        "total_p50_ms": round(statistics.median(r["total_ms"] for r in rows), 1),
    }


def write(name: str, summaries: list[dict], details: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"phase5_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    with open(RESULTS / f"phase5_{name}_details.jsonl", "w", encoding="utf-8") as f:
        for r in details:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in summaries:
        print(s)


def compression(n: int, precision: str) -> None:
    from ncm_lab.models.baselines import HFSeq2Seq
    from ncm_lab.models.ncm import NCMSummarizer

    llm = OllamaLLM()
    questions = eval_questions(n)
    store = VectorStore()
    store.add(paragraphs("en"))
    details, summaries = [], []

    rows = run_questions(RAGPipeline(store, llm), questions, "plain RAG")
    details += rows
    summaries.append(summarize(rows))

    ncm = NCMSummarizer(precision=precision, batch_size=8)
    rows = run_questions(RAGPipeline(store, llm, compressor=ncm.generate), questions,
                         "NCM-compressed (XSum summarizer)")
    details += rows
    summaries.append(summarize(rows))
    del ncm
    free_gpu()

    bart = HFSeq2Seq("xsum", precision=precision, batch_size=8)
    rows = run_questions(RAGPipeline(store, llm, compressor=bart.generate), questions,
                         "AR-compressed (BART-base XSum)")
    details += rows
    summaries.append(summarize(rows))
    del bart
    free_gpu()
    write("rag_compression", summaries, details)


def multilingual(n: int, precision: str) -> None:
    import sacrebleu

    from ncm_lab.models.baselines import HFSeq2Seq
    from ncm_lab.models.ncm import NCMTranslator

    llm = OllamaLLM()
    questions = eval_questions(n)
    de, en = paragraphs("de"), paragraphs("en")
    sentences = [(pid, s) for pid, p in enumerate(de) for s in split_sentences(p)]
    print(f"{len(de)} German paragraphs, {len(sentences)} sentences")
    details, summaries = [], []

    store = VectorStore()
    store.add(en)
    rows = run_questions(RAGPipeline(store, llm), questions, "English originals (upper bound)")
    details += rows
    summaries.append(summarize(rows) | {"translator": "-", "translate_s": 0, "sentences_per_s": 0,
                                        "paragraph_bleu": 100.0})

    for label, make in [("NCM (WMT14)", lambda: NCMTranslator(precision=precision,
                                                                 batch_size=32)),
                        ("opus-mt-de-en (AR)", lambda: HFSeq2Seq("wmt14", precision=precision,
                                                                 batch_size=32))]:
        model = make()
        model.generate([s for _, s in sentences[:32]])  # warm-up
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model.generate([s for _, s in sentences])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        del model
        free_gpu()

        translated = [""] * len(de)
        for (pid, _), s in zip(sentences, out):
            translated[pid] = (translated[pid] + " " + s).strip()
        bleu = sacrebleu.corpus_bleu(translated, [en]).score

        store = VectorStore()
        store.add(translated)
        rows = run_questions(RAGPipeline(store, llm), questions, f"translated by {label}")
        details += rows
        summaries.append(summarize(rows) | {
            "translator": label, "translate_s": round(seconds, 2),
            "sentences_per_s": round(len(sentences) / seconds, 1),
            "paragraph_bleu": round(bleu, 2)})
        (RESULTS / f"phase5_translated_{label.split()[0].lower()}.txt").write_text(
            "\n\n".join(translated[:10]), encoding="utf-8")

    write("rag_multilingual", summaries, details)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["compression", "multilingual"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
    args = ap.parse_args()
    if not OllamaLLM().available():
        raise SystemExit("Ollama is not reachable; start it with: bash scripts/ollama.sh serve")
    {"compression": compression, "multilingual": multilingual}[args.experiment](args.n,
                                                                                 args.precision)


if __name__ == "__main__":
    main()
