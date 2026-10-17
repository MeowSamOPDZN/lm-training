"""
Project-wide device policy: CPU ONLY.

Even though PyTorch is installed with CUDA support (so the same code can
later run on a GPU machine like Colab), every script in this project must
train and run on CPU. This module is the single source of truth for that.
"""

import os

# Belt and suspenders: hide every GPU from PyTorch even if one exists.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

# Hard-forced. Never torch.device("cuda") anywhere in this project.
DEVICE = torch.device("cpu")


def get_device() -> torch.device:
    return DEVICE


def describe_device() -> dict:
    """Facts about the torch build and what it can see."""
    return {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,          # CUDA toolkit torch was built against (None if CPU-only build)
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "selected_device": str(DEVICE),
        "num_threads": torch.get_num_threads(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe_device(), indent=2))
