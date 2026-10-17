"""
Generate text with the GPT model (greedy / temperature / top-k sampling).

Two ways to test:
  1. No checkpoint  -> random weights, expect gibberish. This proves the
     generation machinery (prompt encoding -> sampling loop -> decoding)
     works end to end.
  2. With checkpoint -> run scripts/train.py first, then point at a
     models/checkpoints/step_N.pt file. Output improves with training.

Usage:
    ./venv/bin/python scripts/generate.py --prompt "The capital of France"
    ./venv/bin/python scripts/generate.py --prompt "Once upon a time" \
        --checkpoint models/checkpoints/step_1000.pt --max-new-tokens 200
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_device  # noqa: E402  (CPU-only policy)
from model import GPT, GPTConfig, autocast_ctx  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_PATH = os.path.join(BASE, "models", "tokenizer-gpt2", "tokenizer.json")


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int,
             temperature: float, top_k: int) -> str:
    config = model.config
    device = get_device()
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    if idx.size(1) == 0:
        raise ValueError("prompt encoded to zero tokens")
    with autocast_ctx(config):
        for _ in range(max_new_tokens):
            idx_in = idx[:, -config.block_size:]  # crop to context window
            logits, _ = model(idx_in)
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if nxt.item() == tok.token_to_id("<|endoftext|>"):
                break
    return tok.decode(idx[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--checkpoint", default=None,
                    help="e.g. models/checkpoints/step_1000.pt")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    if not os.path.exists(TOK_PATH):
        print("Tokenizer not found - run scripts/train_tokenizer.py first,")
        print("or git pull (it's in the repo).")
        sys.exit(1)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)

    model = GPT(GPTConfig()).to(get_device())
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=get_device(),
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"loaded {args.checkpoint} (step {ckpt.get('step', '?')})")
    else:
        print("no checkpoint -> random weights, expect gibberish "
              "(this still proves generation works)")
    model.eval()

    out = generate(model, tok, args.prompt, args.max_new_tokens,
                   args.temperature, args.top_k)
    print("-" * 60)
    print(out)
    print("-" * 60)


if __name__ == "__main__":
    main()
