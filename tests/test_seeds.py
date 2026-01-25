"""Unit tests for src.first_ai.seeds module."""

import random
import numpy as np
import pytest

from src.first_ai.seeds import set_global_seed


def test_set_global_seed_basic():
    """Test that set_global_seed sets Python and NumPy seeds."""
    set_global_seed(42, deterministic=False)
    
    # Python random
    py_val1 = random.random()
    set_global_seed(42, deterministic=False)
    py_val2 = random.random()
    assert py_val1 == py_val2, "Python random should be reproducible"
    
    # NumPy random
    set_global_seed(42, deterministic=False)
    np_val1 = np.random.rand()
    set_global_seed(42, deterministic=False)
    np_val2 = np.random.rand()
    assert np_val1 == np_val2, "NumPy random should be reproducible"


def test_set_global_seed_different_values():
    """Test that different seeds produce different values."""
    set_global_seed(42)
    val1 = random.random()
    
    set_global_seed(99)
    val2 = random.random()
    
    assert val1 != val2, "Different seeds should produce different values"


def test_set_global_seed_with_torch():
    """Test torch seeding if available."""
    try:
        import torch
        set_global_seed(42, deterministic=True)
        t1 = torch.rand(1).item()
        set_global_seed(42, deterministic=True)
        t2 = torch.rand(1).item()
        assert t1 == t2, "Torch random should be reproducible"
    except ImportError:
        pytest.skip("torch not available")
