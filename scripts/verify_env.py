"""
Environment verification for the from-scratch LM training setup.

Run with the project venv:
    ~/workspace/lm-training/venv/bin/python scripts/verify_env.py

Checks:
  1. Every required package imports and reports its version.
  2. PyTorch build info: was it built WITH CUDA support?
  3. CUDA detection: is a GPU actually visible right now?
  4. Project device policy: CPU is forced (see scripts/config.py).
  5. A tiny CPU tensor + matmul smoke test.
"""

import importlib
import sys

REQUIRED = [
    "torch",
    "transformers",
    "datasets",
    "tokenizers",
    "numpy",
    "matplotlib",
    "tqdm",
]

failures = []

print("=" * 60)
print("1) Package imports")
print("=" * 60)
versions = {}
for name in REQUIRED:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        versions[name] = ver
        print(f"  [OK] {name:14s} {ver}")
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"  [FAIL] {name:14s} {e!r}")

print()
print("=" * 60)
print("2) PyTorch build info (was it built WITH CUDA support?)")
print("=" * 60)
if "torch" not in failures:
    import torch

    cuda_build = torch.version.cuda
    print(f"  torch version      : {torch.__version__}")
    print(f"  built with CUDA    : {cuda_build if cuda_build else 'NO (CPU-only build)'}")
    print(f"  cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  cuda device count  : {torch.cuda.device_count()}")
    if cuda_build:
        print("  -> CUDA-capable build installed. Same code can use a GPU later.")
    else:
        print("  -> CPU-only build. Reinstall with the CUDA wheel to get GPU support.")

    print()
    print("=" * 60)
    print("3) Project device policy (CPU forced)")
    print("=" * 60)
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    from config import get_device, describe_device  # noqa: E402

    info = describe_device()
    print(f"  selected device    : {info['selected_device']}")
    print(f"  torch threads      : {info['num_threads']}")
    assert str(get_device()) == "cpu", "device policy violated!"

    print()
    print("=" * 60)
    print("4) CPU smoke test")
    print("=" * 60)
    x = torch.randn(256, 256, device=get_device())
    y = x @ x.T
    print(f"  256x256 matmul on {get_device()}: OK (mean={y.mean().item():.4f})")

print()
print("=" * 60)
if failures:
    print(f"RESULT: FAIL - missing/broken: {', '.join(failures)}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED - CPU-only training environment ready.")
