"""Random-source seeding and deterministic-mode switches (section 9).

A fixed seed alone does NOT guarantee reproducibility; the deterministic
algorithms switch, cuDNN benchmark flag, TF32 setting and (for GPU) the
``CUBLAS_WORKSPACE_CONFIG`` env var must all be set explicitly. Their
limitations are listed in section 9.3 and are not removed by this module.
"""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (incl. CUDA) and flip deterministic flags."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # Restrict TF32: it trades bits for speed and breaks bit-identity.
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic

    if deterministic:
        # Required for some cuBLAS deterministic paths on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older PyTorch may not support warn_only; fall back to strict.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def worker_seeds(base_seed: int, count: int) -> list[int]:
    """Deterministic per-worker seed derivation (used by worker_init_fn)."""
    return [(base_seed + i) % (2 ** 31) for i in range(count)]
