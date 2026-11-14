"""Full TinyStories, 256 BPE tokens per story, stored as memory-mapped int16 arrays.

    python -m ncm_lab.scale.data

Uses the Phase 2 BPE (4096 tokens, data/tinystories/bpe.json) so the toy and scaled models
share a vocabulary. The official validation split (about 22k stories) is cut in two:
the first half is `val` (temperature tuning, early stopping), the second half `test`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ncm_lab.toy_ncm.config import EOS_ID, PAD_ID
from ncm_lab.toy_ncm.data import REPO, load_bpe

SEQ_LEN = 256
OUT_DIR = Path("data/tinystories_256")
SPLITS = ("train", "val", "test")


def _parquet_texts(prefix: str) -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"data/{prefix}-") and f.endswith(".parquet"))
    texts: list[str] = []
    for name in files:
        path = hf_hub_download(REPO, name, repo_type="dataset", local_dir=OUT_DIR / "raw")
        texts += [t.strip() for t in pq.read_table(path, columns=["text"]).column("text")
                  .to_pylist()]
    return [t for t in texts if t]


def encode(ids_per_text: list[list[int]], seq_len: int = SEQ_LEN) -> np.ndarray:
    """First seq_len tokens of each text, <eos> if it fits, padded with <pad>."""
    out = np.full((len(ids_per_text), seq_len), PAD_ID, dtype=np.int16)
    for i, ids in enumerate(ids_per_text):
        ids = ids[:seq_len]
        if len(ids) < seq_len:
            ids = ids + [EOS_ID]
        out[i, : len(ids)] = ids
    return out


def load_split(name: str, root: Path = OUT_DIR, in_memory: bool = False) -> np.ndarray:
    """(N, 256) int16. Memory-mapped by default: only the rows a batch touches are read.

    in_memory=True reads the whole file once (1.1 GB for train), which avoids a slow
    random read per batch when the data sits on the Windows drive under WSL.
    """
    return np.load(root / f"{name}.npy", mmap_mode=None if in_memory else "r")


def batch_from(data: np.ndarray, idx: np.ndarray, dev: str) -> torch.Tensor:
    rows = np.sort(idx)  # sorted reads are faster on a memmap; order inside a batch is moot
    return torch.from_numpy(data[rows].astype(np.int64)).to(dev, non_blocking=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bpe = load_bpe()
    valid = _parquet_texts("validation")
    half = len(valid) // 2
    texts = {"train": _parquet_texts("train"), "val": valid[:half], "test": valid[half:]}
    for name in SPLITS:
        arr = np.concatenate([encode([e.ids for e in bpe.encode_batch(texts[name][i:i + 50_000])])
                              for i in range(0, len(texts[name]), 50_000)])
        np.save(OUT_DIR / f"{name}.npy", arr)
        real = int((arr != PAD_ID).sum())
        full = float((arr != PAD_ID).all(1).mean())
        print(f"{name}: {arr.shape[0]:,} stories, {real / 1e6:.1f}M non-pad tokens, "
              f"{full:.0%} fill all {SEQ_LEN}", flush=True)


if __name__ == "__main__":
    main()
