"""Unit tests for src.first_ai.artifacts module."""

import tempfile
from pathlib import Path
import pytest

from src.first_ai.artifacts import (
    resolve_artifact_path, 
    save_artifact, 
    load_artifact,
    ArtifactMeta,
    DEFAULT_MODELS_DIR
)


def test_resolve_artifact_path():
    """Test artifact path resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "models"
        path = resolve_artifact_path("test_model", base_dir=base_dir)
        
        assert path.name == "test_model.pth"
        assert base_dir.exists()


def test_resolve_artifact_path_with_extension():
    """Test that .pth extension is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "models"
        path = resolve_artifact_path("test_model.pth", base_dir=base_dir)
        
        assert path.name == "test_model.pth"


def test_save_and_load_artifact():
    """Test saving and loading artifacts."""
    pytest.importorskip("torch")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "models"
        
        payload = {"model_state": {"weight": [1, 2, 3]}, "epoch": 10}
        meta = ArtifactMeta(name="test_model", version="v1.0", notes="test")
        
        path = save_artifact(payload, "test_model", meta=meta, base_dir=base_dir)
        assert path.exists()
        
        loaded = load_artifact("test_model", base_dir=base_dir)
        assert "payload" in loaded
        assert loaded["payload"]["epoch"] == 10
        assert "meta" in loaded
        assert loaded["meta"]["version"] == "v1.0"


def test_save_artifact_without_meta():
    """Test saving artifact without metadata."""
    pytest.importorskip("torch")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "models"
        
        payload = {"data": "test"}
        path = save_artifact(payload, "simple", meta=None, base_dir=base_dir)
        
        loaded = load_artifact("simple", base_dir=base_dir)
        assert loaded["payload"]["data"] == "test"
        assert "meta" not in loaded
