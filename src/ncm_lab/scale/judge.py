"""Phase 10 coherence judge: qwen2.5:3b (Ollama) rates stories 1-5, blind.

    PYTHONPATH=src python -m ncm_lab.scale.judge items     # WSL: build the blind item list
    python -m ncm_lab.scale.judge score                    # Windows: ask Ollama, resumable
    python -m ncm_lab.scale.judge summary

Items: the first N samples of every model in results/phase10_samples.jsonl, N real test
stories, and the same N stories with their sentences shuffled. The judge must rate the real
stories clearly above the shuffled ones, or its scores mean nothing. Items are shuffled and
carry only an id, so the judge never sees which model wrote what. `score` uses only the
standard library, so it runs where Ollama runs (Windows) without the project's GPU stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import urllib.request
from pathlib import Path

RESULTS = Path("results")
ITEMS = RESULTS / "phase10_judge_items.jsonl"
SCORES = RESULTS / "phase10_judge_scores.jsonl"
SUMMARY = RESULTS / "phase10_judge.csv"
N = 200
MODEL = "qwen2.5:3b"
URL = "http://localhost:11434/api/generate"
PROMPT = """You rate short children's stories for coherence.
Score from 1 to 5:
1 = word salad, no sense at all
2 = mostly broken: some phrases make sense, sentences do not
3 = understandable, but with clear grammar or logic errors
4 = coherent, with minor errors
5 = fully coherent and fluent
Reply with the number only.

Story:
{story}

Score:"""


def shuffle_sentences(text: str, rng: random.Random) -> str:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    rng.shuffle(sents)
    return " ".join(sents)


def build_items(n: int = N) -> list[dict]:
    from .evaluate import N_TEST, DRAWS, load_bpe, reference_texts

    per_run: dict[str, list[str]] = {}
    with open(RESULTS / "phase10_samples.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            per_run.setdefault(d["run"], []).append(d["text"])
    # human stories outside the MAUVE reference draws (those used the first N_TEST * DRAWS)
    human = reference_texts("test", N_TEST * DRAWS + n, load_bpe())[-n:]
    rng = random.Random(0)
    groups = {run: texts[:n] for run, texts in per_run.items()}
    groups["human"] = human
    groups["human-shuffled"] = [shuffle_sentences(t, rng) for t in human]
    items = [{"group": g, "text": t} for g, texts in groups.items() for t in texts]
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["id"] = i
    return items


def ask(story: str) -> int | None:
    body = json.dumps({"model": MODEL, "prompt": PROMPT.format(story=story), "stream": False,
                       "options": {"temperature": 0, "num_predict": 4, "seed": 0}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.loads(r.read())["response"]
    m = re.search(r"[1-5]", reply)
    return int(m.group()) if m else None


def score() -> None:
    items = [json.loads(line) for line in open(ITEMS, encoding="utf-8")]
    done = {json.loads(line)["id"] for line in open(SCORES, encoding="utf-8")} \
        if SCORES.exists() else set()
    with open(SCORES, "a", encoding="utf-8") as out:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            out.write(json.dumps({"id": it["id"], "group": it["group"],
                                  "score": ask(it["text"])}) + "\n")
            out.flush()
            if i % 100 == 0:
                print(f"{i}/{len(items)}", flush=True)


def summary() -> list[dict]:
    by: dict[str, list[int]] = {}
    unparsed: dict[str, int] = {}
    for line in open(SCORES, encoding="utf-8"):
        d = json.loads(line)
        if d["score"] is None:
            unparsed[d["group"]] = unparsed.get(d["group"], 0) + 1
        else:
            by.setdefault(d["group"], []).append(d["score"])
    rows = [{"group": g, "n": len(s), "mean": round(statistics.fmean(s), 3),
             "sd": round(statistics.stdev(s), 3) if len(s) > 1 else 0.0,
             "share_4_or_5": round(sum(x >= 4 for x in s) / len(s), 3),
             "unparsed": unparsed.get(g, 0)}
            for g, s in sorted(by.items(), key=lambda kv: -statistics.fmean(kv[1]))]
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["items", "score", "summary"])
    a = p.parse_args(argv)
    if a.cmd == "items":
        items = build_items()
        with open(ITEMS, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"{len(items)} items")
    elif a.cmd == "score":
        score()
    else:
        summary()


if __name__ == "__main__":
    main()
