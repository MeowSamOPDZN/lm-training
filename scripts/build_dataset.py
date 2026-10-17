"""
Encode the raw corpora into token-id sequences for LM training.

- Loads models/tokenizer-gpt2/tokenizer.json
- Splits each raw file on <|endoftext|>, encodes docs, appends EOS
- Token-level 80/20 train/val split (val = last 20% of tokens)
- Chunks into fixed block_size sequences, drops the tail remainder
- Saves data/tokenized/train.bin + val.bin as uint16 memmap + meta.json

Usage:
    ~/workspace/lm-training/venv/bin/python scripts/build_dataset.py [--block-size 1024]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PATH = os.path.join(BASE, "models", "tokenizer-gpt2", "tokenizer.json")
RAW_GLOB = os.path.join(BASE, "data", "raw", "*.txt")
OUT_DIR = os.path.join(BASE, "data", "tokenized")
EOS = "<|endoftext|>"
VAL_FRACTION = 0.20  # 80/20 train/validation split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=1024)
    args = ap.parse_args()

    if not os.path.exists(TOK_PATH):
        print("Tokenizer not found - run scripts/train_tokenizer.py first.")
        sys.exit(1)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOK_PATH)
    eos_id = tok.token_to_id(EOS)
    assert eos_id is not None, "EOS token missing from tokenizer"
    vocab_size = tok.get_vocab_size()
    assert vocab_size < 65536, "vocab too large for uint16 storage"

    files = sorted(glob.glob(RAW_GLOB))
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        print("No raw .txt files in data/raw/.")
        sys.exit(1)

    all_ids: list[int] = []
    n_docs = 0
    for path in files:
        name = os.path.basename(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            docs = [d.strip() for d in f.read().split(EOS)]
        docs = [d for d in docs if d]
        print(f"{name}: {len(docs)} docs, encoding...", flush=True)
        # batch encode for speed
        for i in range(0, len(docs), 512):
            batch = docs[i:i + 512]
            for enc in tok.encode_batch(batch):
                all_ids.extend(enc.ids)
                all_ids.append(eos_id)
            n_docs += len(batch)
            if (i // 512) % 20 == 0:
                print(f"  ... {n_docs} docs, {len(all_ids)/1e6:.1f}M tokens",
                      flush=True)

    arr = np.array(all_ids, dtype=np.uint16)
    n_total = len(arr)
    n_val = int(n_total * VAL_FRACTION)
    n_val -= n_val % args.block_size  # keep val a whole number of blocks
    n_train = n_total - n_val
    n_train -= n_train % args.block_size

    os.makedirs(OUT_DIR, exist_ok=True)
    train = arr[:n_train].reshape(-1, args.block_size)
    val = arr[n_train:n_train + n_val].reshape(-1, args.block_size)

    def save_memmap(a: np.ndarray, name: str) -> None:
        path = os.path.join(OUT_DIR, name)
        fp = np.memmap(path, dtype=np.uint16, mode="w+", shape=a.shape)
        fp[:] = a[:]
        fp.flush()
        del fp

    save_memmap(train, "train.bin")
    save_memmap(val, "val.bin")

    meta = {
        "vocab_size": vocab_size,
        "eos_token": EOS,
        "eos_id": eos_id,
        "block_size": args.block_size,
        "n_train_tokens": int(n_train),
        "n_val_tokens": int(n_val),
        "n_train_seqs": int(train.shape[0]),
        "n_val_seqs": int(val.shape[0]),
        "dtype": "uint16",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n_total/1e6:.1f}M tokens from {n_docs} docs "
          f"(vocab {vocab_size})")
    print(f"  train: {train.shape[0]} seqs x {args.block_size} "
          f"({n_train/1e6:.1f}M tokens)")
    print(f"  val:   {val.shape[0]} seqs x {args.block_size} "
          f"({n_val/1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
