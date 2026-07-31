"""Deterministic seeding across numpy, torch and sklearn (P4)."""
from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> dict:
    """Seed every RNG that can affect a reported number.

    Returns a record of what was seeded, and of any residual nondeterminism
    that could not be eliminated -- cuDNN kernel selection in particular is
    documented rather than silently ignored (P4).
    """
    rec: dict = {"seed": seed, "nondeterminism": []}
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        rec["numpy"] = True
    except Exception:
        rec["numpy"] = False
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        rec["torch"] = True
        rec["nondeterminism"].append(
            "cuDNN set to deterministic and benchmark disabled; some fp16 reductions "
            "on GPU remain order-dependent, so embedding values may differ in the last "
            "bits between runs. Downstream metrics are computed from cached embeddings, "
            "so the sweep and statistics ARE bit-reproducible from a fixed cache."
        )
    except Exception:
        rec["torch"] = False
    return rec
