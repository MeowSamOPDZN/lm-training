"""
Train the GPT model on the tokenized dataset.

Wires together:
  scripts/data_loader.py  -> batched (x, y) next-token pairs, 80/20 split
  scripts/model.py        -> GPT, AdamW, CPU bfloat16 autocast,
                             gradient checkpointing

Checkpoints go to models/checkpoints/step_*.pt (model + optimizer state).

Usage:
    ~/workspace/lm-training/venv/bin/python scripts/train.py [--steps 1000] [--batch-size 4]
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_device  # noqa: E402
from data_loader import get_loaders  # noqa: E402
from model import GPT, GPTConfig, autocast_ctx, bf16_supported  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE, "models", "checkpoints")


def lr_schedule(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    device = get_device()
    config = GPTConfig()
    model = GPT(config).to(device)
    print(f"params: {model.num_parameters() / 1e6:.2f}M | device: {device} | "
          f"bf16: {bf16_supported() and config.use_mixed_precision} | "
          f"checkpointing: {config.use_checkpointing}")

    train_loader, val_loader = get_loaders(batch_size=args.batch_size)
    opt = model.configure_optimizers(lr=args.lr)
    os.makedirs(CKPT_DIR, exist_ok=True)

    model.train()
    t0 = time.time()
    train_iter = iter(train_loader)
    for step in range(args.steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        for pg in opt.param_groups:
            pg["lr"] = lr_schedule(step, args.warmup, args.steps, args.lr)

        opt.zero_grad()
        with autocast_ctx(config):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            toks = args.log_every * args.batch_size * config.block_size
            print(f"step {step + 1:5d}/{args.steps} | loss {loss.item():.4f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | "
                  f"{toks / dt:,.0f} tok/s", flush=True)
            t0 = time.time()

        if (step + 1) % args.ckpt_every == 0:
            path = os.path.join(CKPT_DIR, f"step_{step + 1}.pt")
            torch.save({"step": step + 1,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "config": config.__dict__}, path)
            print(f"  saved {path}")

    # quick validation pass
    model.eval()
    val_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with autocast_ctx(config):
                _, loss = model(x, y)
            val_loss += loss.item()
            n += 1
            if n >= 20:
                break
    print(f"val loss (20 batches): {val_loss / n:.4f}")


if __name__ == "__main__":
    main()
