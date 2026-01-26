from __future__ import annotations

"""Artifact helpers for saving/loading model payloads.

Used by training scripts and detection utilities via Config.MODELS_DIR.
Requires torch availability for serialization.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except Exception:  # torch may be absent in some environments
    torch = None  # type: ignore


DEFAULT_MODELS_DIR = Path("models")


@dataclass
class ArtifactMeta:
    name: str
    version: Optional[str] = None
    notes: Optional[str] = None


def resolve_artifact_path(name: str, base_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".pth"):
        name = f"{name}.pth"
    return base_dir / name


def save_artifact(payload: Dict[str, Any], name: str, meta: Optional[ArtifactMeta] = None, base_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """
    Save a dict payload (e.g., model_state + metadata) to models directory.

    Args:
        payload: Arbitrary dict to serialize via torch.save.
        name: Base filename (without extension or with .pth).
        meta: Optional metadata to include.
        base_dir: Base directory for artifacts (defaults to DEFAULT_MODELS_DIR).

    Returns:
        Path to the saved artifact.
    """
    path = resolve_artifact_path(name, base_dir)
    envelope = {"payload": payload}
    if meta is not None:
        envelope["meta"] = asdict(meta)

    if torch is None:
        raise RuntimeError("torch not available for saving artifacts")
    with open(path, "wb") as f:
        torch.save(envelope, f)
    return path


def load_artifact(name: str, base_dir: Path = DEFAULT_MODELS_DIR) -> Dict[str, Any]:
    """Load an artifact envelope and return its payload+meta dict."""
    path = resolve_artifact_path(name, base_dir)
    if torch is None:
        raise RuntimeError("torch not available for loading artifacts")
    with open(path, "rb") as f:
        obj = torch.load(f)
    return obj
