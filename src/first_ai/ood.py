"""OOD computation utilities (scaffold)."""

from typing import Dict, Any

try:
    import torch
    import numpy as np
except Exception:
    torch = None  # type: ignore
    np = None  # type: ignore


def compute_thresholds(distances, percentile: int = 95) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy not available")
    return {
        "percentile": percentile,
        "threshold": float(np.percentile(distances, percentile)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
    }
