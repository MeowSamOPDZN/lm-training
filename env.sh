# Activate with: source ~/workspace/lm-training/venv/bin/activate
# Or call the venv python directly: ~/workspace/lm-training/venv/bin/python ...

# Hide GPUs from every tool in this project (CPU-only policy).
export CUDA_VISIBLE_DEVICES=""

# Sensible CPU threading for training on this box.
export OMP_NUM_THREADS="$(nproc)"
export MKL_NUM_THREADS="$(nproc)"

# Hugging Face cache inside the project (keeps home dir clean).
export HF_HOME="$HOME/workspace/lm-training/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
