"""
Encode the raw corpora into token-id sequences for LM training.

Memory-safe: token ids are streamed to a temporary flat uint16 file
(never held in RAM all at once), then split and copied into the
final memmaps in chunks.

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
TMP_PATH = os.path.join(OUT_DIR, "_all_tokens.u16")
EOS = "<|endoftext|>"
VAL_FRACTION = 0.20  # 80/20 train/validation split
CHUNK_TOKENS = 5_000_000  # memmap copy chunk size (keeps RAM low)


def iter_docs(path: str, sep: bytes = EOS.encode("utf-8")):
    """Yield documents split on the EOS separator, streaming in 8MB blocks.

    The separator is pure ASCII so byte-level splitting is UTF-8 safe.
    Never holds more than ~16MB in RAM regardless of file size.
    """
    with open(path, "rb") as f:
        buf = b""
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(sep)
            buf = parts.pop()  # last piece may straddle the next block
            for p in parts:
                text = p.decode("utf-8", errors="ignore").strip()
                if text:
                    yield text
        tail = buf.decode("utf-8", errors="ignore").strip()
        if tail:
            yield tail


def batched(iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


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

    os.makedirs(OUT_DIR, exist_ok=True)

    # Phase 1: stream-encode -> flat temp file, counting as we go.
    # Files are read in 8MB blocks (iter_docs) so RAM stays flat
    # no matter how large the corpus is.
    n_docs, n_total = 0, 0
    with open(TMP_PATH, "wb") as tmp:
        for path in files:
            name = os.path.basename(path)
            print(f"{name}: encoding...", flush=True)
            for bi, batch in enumerate(batched(iter_docs(path), 256)):
                buf = []
                for enc in tok.encode_batch(batch):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                np.array(buf, dtype=np.uint16).tofile(tmp)
                n_docs += len(batch)
                n_total += len(buf)
                if bi % 40 == 0:
                    print(f"  ... {n_docs} docs, {n_total / 1e6:.1f}M tokens",
                          flush=True)
    print(f"encoded {n_total / 1e6:.1f}M tokens from {n_docs} docs")

    # Phase 2: split 80/20 and copy into block-aligned memmaps, in chunks.
    flat = np.memmap(TMP_PATH, dtype=np.uint16, mode="r", shape=(n_total,))
    n_val = int(n_total * VAL_FRACTION)
    n_val -= n_val % args.block_size
    n_train = n_total - n_val
    n_train -= n_train % args.block_size

    def copy_slice(dst_path: str, start: int, length: int) -> None:
        n_seqs = length // args.block_size
        fp = np.memmap(dst_path, dtype=np.uint16, mode="w+",
                       shape=(n_seqs, args.block_size))
        done = 0
        while done < length:
            n = min(CHUNK_TOKENS, length - done)
            fp.reshape(-1)[done:done + n] = flat[start + done:start + done + n]
            done += n
        fp.flush()
        del fp

    copy_slice(os.path.join(OUT_DIR, "train.bin"), 0, n_train)
    copy_slice(os.path.join(OUT_DIR, "val.bin"), n_train, n_val)
    del flat
    os.remove(TMP_PATH)

    meta = {
        "vocab_size": vocab_size,
        "eos_token": EOS,
        "eos_id": eos_id,
        "block_size": args.block_size,
        "n_train_tokens": int(n_train),
        "n_val_tokens": int(n_val),
        "n_train_seqs": int(n_train // args.block_size),
        "n_val_seqs": int(n_val // args.block_size),
        "dtype": "uint16",
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {n_total / 1e6:.1f}M tokens from {n_docs} docs "
          f"(vocab {vocab_size})")
    print(f"  train: {meta['n_train_seqs']} seqs x {args.block_size} "
          f"({n_train / 1e6:.1f}M tokens)")
    print(f"  val:   {meta['n_val_seqs']} seqs x {args.block_size} "
          f"({n_val / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    main()
