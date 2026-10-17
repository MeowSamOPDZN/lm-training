"""
PyTorch DataLoader over the tokenized memmap dataset.

Each item is a block of `block_size` token ids; the loader returns
(x, y) where y is x shifted one position left (next-token prediction).

Usage:
    from data_loader import get_loaders
    train_loader, val_loader = get_loaders(batch_size=8)
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.path.join(BASE, "data", "tokenized")

sys.path.insert(0, os.path.join(BASE, "scripts"))
from config import get_device  # noqa: E402  (CPU-only policy)


class MemmapLMDataset(Dataset):
    def __init__(self, split: str = "train"):
        assert split in ("train", "val")
        with open(os.path.join(TOK_DIR, "meta.json")) as f:
            self.meta = json.load(f)
        path = os.path.join(TOK_DIR, f"{split}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing - run scripts/build_dataset.py first")
        n_seqs = self.meta[f"n_{split}_seqs"]
        block = self.meta["block_size"]
        self.data = np.memmap(path, dtype=np.uint16, mode="r",
                              shape=(n_seqs, block))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data[idx].astype(np.int64))
        y = torch.roll(x, shifts=-1)  # next-token targets
        return x, y


def get_loaders(batch_size: int = 8, num_workers: int = 0):
    """(train_loader, val_loader). num_workers=0: memmap is fork-sensitive."""
    train_ds = MemmapLMDataset("train")
    val_ds = MemmapLMDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True)
    return train_loader, val_loader


if __name__ == "__main__":
    print("device policy:", get_device())
    train_loader, val_loader = get_loaders(batch_size=4)
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")
    x, y = next(iter(train_loader))
    print("x:", x.shape, x.dtype, "| y:", y.shape)
    assert torch.equal(y[:, :-1], x[:, 1:]), "shift mismatch!"
    print("shift check OK: y[t] == x[t+1]")

    # show sample tokenized outputs to verify the pipeline
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(
        os.path.join(BASE, "models", "tokenizer-gpt2", "tokenizer.json"))
    print("\n--- sample token ids (first 30 of batch 0) ---")
    print(x[0][:30].tolist())
    print("\n--- decoded back to text (first 400 chars) ---")
    print(tok.decode(x[0][:80].tolist())[:400])
