"""Shared settings for the toy NCM (Phase 2)."""

from pathlib import Path

DATA_DIR = Path("data/tinystories")
ART_DIR = Path("artifacts/toy_ncm")
RESULTS_DIR = Path("results")

SEQ_LEN = 64            # tokens per example (first 64 BPE tokens of a story)
VOCAB_SIZE = 4096       # BPE vocabulary trained on TinyStories
PAD_ID = 0
EOS_ID = 1              # also used as the start token of the AR baseline

LEVELS = (1, 4, 8, 16, 32, 64)  # length of each level, coarse to fine
N_CODES = 512           # codebook size per level
CODE_DIM = 32           # dimension of the quantized vectors

DIM = 256               # transformer width for every model
HEADS = 4
BATCH = 128


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
