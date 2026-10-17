"""
GPT-style decoder-only transformer, from scratch.

Config here = GPT-2 small scale: 12 layers, 768 hidden, 12 heads,
~124M parameters (within the 100-200M target). Components:
  - token + learned positional embeddings (weights tied: wte == lm_head)
  - pre-norm transformer blocks: LayerNorm -> causal multi-head
    self-attention -> LayerNorm -> feed-forward (GELU, 4x expansion)
  - final LayerNorm + linear head, next-token cross-entropy loss
  - AdamW optimizer with GPT-style decay / no-decay parameter groups
  - mixed precision via CPU bfloat16 autocast (falls back to fp32)
  - gradient checkpointing per block (torch.utils.checkpoint)

Everything is CPU-forced per scripts/config.py.

Usage:
    ~/workspace/lm-training/venv/bin/python scripts/model.py   # summary + smoke test
"""

import contextlib
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_device  # noqa: E402  (CPU-only policy)


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024       # max context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True            # biases in LayerNorm + Linear
    use_checkpointing: bool = True
    use_mixed_precision: bool = True


# ----------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (lower-triangular) mask."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must divide n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward: n_embd -> 4*n_embd -> GELU -> n_embd."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ----------------------------------------------------------------------------
# full model
# ----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # position
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying (GPT-2): input embeddings == output head
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence longer than block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            if self.config.use_checkpointing and self.training:
                # recompute activations on backward: less RAM, more compute
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:  # next-token prediction
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    def configure_optimizers(self, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1):
        """AdamW; weight decay on 2D weights only (GPT-style groups)."""
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# mixed precision (CPU bfloat16 autocast, fp32 fallback)
# ----------------------------------------------------------------------------
def bf16_supported() -> bool:
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            (torch.randn(8, 8) @ torch.randn(8, 8)).float()
        return True
    except Exception:
        return False


def autocast_ctx(config: GPTConfig):
    if config.use_mixed_precision and bf16_supported():
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ----------------------------------------------------------------------------
# summary + smoke test
# ----------------------------------------------------------------------------
def model_summary(model: GPT) -> None:
    cfg = model.config
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"  architecture : GPT-style decoder-only transformer")
    print(f"  layers       : {cfg.n_layer}")
    print(f"  hidden dim   : {cfg.n_embd}")
    print(f"  attn heads   : {cfg.n_head}")
    print(f"  head dim     : {cfg.n_embd // cfg.n_head}")
    print(f"  ffn dim      : {4 * cfg.n_embd} (GELU)")
    print(f"  context      : {cfg.block_size} tokens")
    print(f"  vocab        : {cfg.vocab_size} (embeddings tied to lm_head)")
    print(f"  norm         : LayerNorm (pre-norm blocks + final)")
    print(f"  checkpointing: {'on' if cfg.use_checkpointing else 'off'}")
    print(f"  mixed prec.  : "
          f"{'bfloat16' if cfg.use_mixed_precision and bf16_supported() else 'fp32'}")
    print("-" * 60)
    parts = [("token+pos embeddings", [model.wte, model.wpe]),
             ("attention blocks", [model.blocks]),
             ("final norm (head tied to wte)", [model.ln_f])]
    total = 0
    for name, mods in parts:
        n = sum(p.numel() for m in mods for p in m.parameters())
        total += n
        print(f"  {name:28s} {n / 1e6:8.2f}M params")
    print("-" * 60)
    print(f"  TOTAL                    {total / 1e6:8.2f}M params")
    print(f"  model.num_parameters {model.num_parameters() / 1e6:8.2f}M params")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = get_device()
    config = GPTConfig()
    model = GPT(config).to(device)
    model_summary(model)

    # smoke test: one forward + backward on random tokens, exactly like
    # train.py does (autocast + checkpointing + AdamW step)
    model.train()
    opt = model.configure_optimizers(lr=3e-4)
    B, T = 2, 128
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    y = torch.randint(0, config.vocab_size, (B, T), device=device)
    opt.zero_grad()
    with autocast_ctx(config):
        logits, loss = model(x, y)
    print(f"\nforward: logits {tuple(logits.shape)}, loss {loss.item():.4f}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print("backward + AdamW step: OK "
          f"(checkpointing={'on' if config.use_checkpointing else 'off'})")
