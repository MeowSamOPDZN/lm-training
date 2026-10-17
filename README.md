# LM Training — from-scratch language model workspace

CPU-only training environment. PyTorch is installed as the CUDA-capable
build (so this exact code can later run on a GPU box like Colab), but the
project policy forces CPU everywhere. There is no GPU on this machine and
`CUDA_VISIBLE_DEVICES=""` hides one even if it appeared.

## Layout

```
lm-training/
  venv/        isolated Python environment (do not commit)
  data/        training / eval corpora go here
  models/      checkpoints and final weights go here
  scripts/     training & utility scripts
    config.py      single source of truth: DEVICE = cpu, always
    verify_env.py  environment check (run after setup / on new machines)
  env.sh       shell env: hides GPUs, sets thread counts, HF cache dir
```

## Setup (fresh machine)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets tokenizers numpy matplotlib tqdm
source env.sh
python scripts/verify_env.py
```

## Verify

```bash
~/workspace/lm-training/venv/bin/python ~/workspace/lm-training/scripts/verify_env.py
```

Expected: all packages import, torch reports a CUDA build
(`torch.version.cuda` set) while `torch.cuda.is_available()` is `False`,
selected device is `cpu`, and the matmul smoke test passes.

## Data pipeline (raw text -> DataLoader)

```bash
# 1. download corpora (Simple Wikipedia dump + Gutenberg books)
./venv/bin/python scripts/download_books.py
# (wiki dump: curl -C - -o data/raw/simplewiki.xml.bz2 <dump url>)

# 2. parse wiki XML -> clean text
./venv/bin/python scripts/parse_wiki.py

# 3. train GPT-2-style tokenizer (byte-level BPE, 50257 vocab)
./venv/bin/python scripts/train_tokenizer.py

# 4. encode + chunk into 1024-token blocks, 80/20 train/val memmap
./venv/bin/python scripts/build_dataset.py

# 5. sanity check: batches, shift, sample token ids + decoded text
./venv/bin/python scripts/data_loader.py
```

## Model (GPT-style, 124M)

`scripts/model.py`: 12 layers, 768 hidden, 12 heads, 1024 context,
50257 vocab with tied input/output embeddings. Pre-norm blocks
(LayerNorm -> causal multi-head attention -> LayerNorm -> GELU FFN),
GPT-2 weight init, AdamW with decay/no-decay groups, next-token
cross-entropy loss. CPU bfloat16 autocast (fp32 fallback) and per-block
gradient checkpointing are on by default.

```bash
./venv/bin/python scripts/model.py   # summary + forward/backward smoke test
```

## Training

```bash
./venv/bin/python scripts/train.py --steps 1000 --batch-size 4
```

Cosine LR schedule with warmup, gradient clipping, checkpoints to
`models/checkpoints/`, quick 20-batch validation pass at the end.

## Rules

- Never call `torch.device("cuda")` — import `DEVICE` / `get_device()`
  from `scripts/config.py` instead.
- Keep datasets under `data/`, checkpoints under `models/`.
- This project trains on CPU. For GPU runs (e.g. Colab T4), copy the
  scripts and flip the device policy there — do not change it here.
