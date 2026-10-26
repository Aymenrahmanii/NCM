"""Small vector store: sentence-transformers embeddings + a FAISS inner-product index."""

from dataclasses import dataclass

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, runs fine on CPU


@dataclass
class Hit:
    text: str
    doc_id: int
    score: float


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        self.texts: list[str] = []
        self.doc_ids: list[int] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    def add(self, texts: list[str], doc_ids: list[int] | None = None) -> None:
        self.index.add(self._embed(texts))
        if doc_ids is None:
            doc_ids = list(range(len(self.texts), len(self.texts) + len(texts)))
        self.doc_ids += doc_ids
        self.texts += texts

    def search(self, query: str, k: int = 3) -> list[Hit]:
        scores, idx = self.index.search(self._embed([query]), k)
        return [Hit(self.texts[i], self.doc_ids[i], float(s))
                for s, i in zip(scores[0], idx[0]) if i >= 0]

    def __len__(self) -> int:
        return len(self.texts)
