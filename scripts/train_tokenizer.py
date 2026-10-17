"""
Train a GPT-2-style tokenizer: byte-level BPE, exactly the algorithm
GPT-2 uses (pretokenization into bytes, BPE merges learned on text).

Trains on data/raw/*.txt and saves to models/tokenizer-gpt2/:
  tokenizer.json  (fast tokenizer, load with PreTrainedTokenizerFast)
  vocab.json + merges.txt  (GPT-2 file layout)

Usage:
    ~/workspace/lm-training/venv/bin/python scripts/train_tokenizer.py [--vocab-size 50257]
"""

import argparse
import glob
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_GLOB = os.path.join(BASE, "data", "raw", "*.txt")
OUT_DIR = os.path.join(BASE, "models", "tokenizer-gpt2")
EOS = "<|endoftext|>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=50257,
                    help="GPT-2 uses 50257")
    args = ap.parse_args()

    from tokenizers import ByteLevelBPETokenizer

    files = sorted(glob.glob(RAW_GLOB))
    files = [f for f in files if os.path.getsize(f) > 0]
    if not files:
        print("No raw .txt files in data/raw/. Download/parse data first.")
        sys.exit(1)
    print("Training files:", [os.path.basename(f) for f in files])

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    tok.save_model(OUT_DIR)  # vocab.json + merges.txt
    tok.save(os.path.join(OUT_DIR, "tokenizer.json"))

    eos_id = tok.token_to_id(EOS)
    print(f"Saved to {OUT_DIR} | vocab={tok.get_vocab_size()} | "
          f"{EOS} id={eos_id}")

    # round-trip sanity check
    sample = "Hello world! The quick brown fox jumps over 2 lazy dogs. <|endoftext|> Hi."
    ids = tok.encode(sample).ids
    back = tok.decode(ids)
    print("encode:", ids[:20], "...")
    print("roundtrip ok:", back == sample)
    print("decode sample:", tok.decode(ids[:12]))


if __name__ == "__main__":
    main()
