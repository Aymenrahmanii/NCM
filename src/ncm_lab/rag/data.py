"""XQuAD: the same Wikipedia paragraphs and human-written questions in English and German.

English questions with known answers drive every RAG experiment; the German paragraphs feed
the multilingual pipeline.
"""

import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path("data/xquad")
REPO = "google/xquad"


@lru_cache
def load_xquad(lang: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    name = next(f for f in list_repo_files(REPO, repo_type="dataset")
                if f.startswith(f"xquad.{lang}/") and f.endswith(".parquet"))
    path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=DATA_DIR)
    return pq.read_table(path).to_pylist()


def paragraphs(lang: str) -> list[str]:
    """Unique paragraphs, in the same order for every language."""
    seen, out = set(), []
    for row in load_xquad(lang):
        if row["context"] not in seen:
            seen.add(row["context"])
            out.append(row["context"])
    return out


def eval_questions(n: int = 40) -> list[dict]:
    """n English questions spread over the corpus, each with gold answers and paragraph id."""
    rows = load_xquad("en")
    ctx_id = {c: i for i, c in enumerate(paragraphs("en"))}
    first_per_paragraph = {}
    for row in rows:
        first_per_paragraph.setdefault(ctx_id[row["context"]], row)
    ids = sorted(first_per_paragraph)
    step = max(1, len(ids) // n)
    picked = [first_per_paragraph[i] for i in ids[::step][:n]]
    return [{"id": r["id"], "question": r["question"], "answers": list(r["answers"]["text"]),
             "paragraph": ctx_id[r["context"]]} for r in picked]


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter, good enough for Wikipedia prose."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\"„])", text.strip())
    return [p for p in parts if p]


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation, articles and extra spaces."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_correct(prediction: str, answers: list[str]) -> bool:
    """Lenient match: a gold answer appears inside the prediction."""
    pred = normalize(prediction)
    return any(normalize(a) and normalize(a) in pred for a in answers)
