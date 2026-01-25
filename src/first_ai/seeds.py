"""Seed helpers for reproducibility across Python/NumPy/Torch.

Imported by the first_ai CLI and training scripts to enforce deterministic
runs where possible.
"""

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # torch may be absent in some environments
    torch = None  # type: ignore


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set global seeds for reproducibility across Python, NumPy, and Torch.

    Args:
        seed: Seed value to set.
        deterministic: If True and torch is available, set cuDNN deterministic flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                # Backends may not exist depending on Torch build
                pass
