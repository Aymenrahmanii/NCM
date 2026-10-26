"""TinyStories download, BPE training and fixed-length encoding.

    python -m ncm_lab.toy_ncm.data
"""

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import DATA_DIR, EOS_ID, PAD_ID, SEQ_LEN, VOCAB_SIZE

REPO = "roneneldan/TinyStories"
N_TRAIN = 300_000
N_TEST = 2_000
N_BPE = 100_000


def _read_parquet_texts(prefix: str) -> list[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(
        f for f in list_repo_files(REPO, repo_type="dataset")
        if f.startswith(f"data/{prefix}-") and f.endswith(".parquet")
    )
    path = hf_hub_download(REPO, files[0], repo_type="dataset", local_dir=DATA_DIR / "raw")
    return [t.strip() for t in pq.read_table(path, columns=["text"]).column("text").to_pylist()]


def train_bpe(texts: list[str]) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<eos>"],  # ids 0 and 1
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def encode_fixed(tok: Tokenizer, texts: list[str]) -> torch.Tensor:
    """First SEQ_LEN tokens of each text, <eos> if it fits, padded with <pad>."""
    rows = []
    for enc in tok.encode_batch(texts):
        ids = enc.ids[:SEQ_LEN]
        if len(ids) < SEQ_LEN:
            ids = ids + [EOS_ID]
        rows.append(ids + [PAD_ID] * (SEQ_LEN - len(ids)))
    return torch.tensor(rows, dtype=torch.int16)


def decode_ids(tok: Tokenizer, ids) -> str:
    ids = [int(i) for i in ids]
    for stop in (EOS_ID, PAD_ID):
        if stop in ids:
            ids = ids[: ids.index(stop)]
    return tok.decode(ids)


def load_bpe() -> Tokenizer:
    return Tokenizer.from_file(str(DATA_DIR / "bpe.json"))


def load_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.load(DATA_DIR / "tokens.pt")
    return d["train"].long(), d["test"].long()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_texts = _read_parquet_texts("train")[:N_TRAIN]
    test_texts = _read_parquet_texts("validation")[:N_TEST]
    print(f"stories: train {len(train_texts)}, test {len(test_texts)}")

    tok = train_bpe(train_texts[:N_BPE])
    tok.save(str(DATA_DIR / "bpe.json"))

    train, test = encode_fixed(tok, train_texts), encode_fixed(tok, test_texts)
    torch.save({"train": train, "test": test}, DATA_DIR / "tokens.pt")
    full = (train != PAD_ID).all(1).float().mean().item()
    print(f"saved {tuple(train.shape)} train, {tuple(test.shape)} test; "
          f"{full:.0%} use all 64 tokens")
    print("example:", decode_ids(tok, test[0]))


if __name__ == "__main__":
    main()
