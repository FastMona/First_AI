"""Unit tests for detection_utils module."""

import pytest

# Skip entire module if torch not available
pytest.importorskip("torch")

from detection_utils import parse_filename


def test_parse_filename_digit():
    """Test parsing filenames with digit labels."""
    is_digit, label = parse_filename("img_3.jpg")
    assert is_digit is True
    assert label == 3
    
    is_digit, label = parse_filename("img_0.png")
    assert is_digit is True
    assert label == 0
    
    is_digit, label = parse_filename("img_9.jpeg")
    assert is_digit is True
    assert label == 9


def test_parse_filename_ood():
    """Test parsing filenames with non-digit labels (OOD)."""
    is_digit, label = parse_filename("img_a.jpg")
    assert is_digit is False
    assert label is None
    
    is_digit, label = parse_filename("img_cat.png")
    assert is_digit is False
    assert label is None
    
    is_digit, label = parse_filename("img_letter_x.jpg")
    assert is_digit is False
    assert label is None


def test_parse_filename_multi_digit():
    """Test that multi-digit numbers are treated as OOD."""
    is_digit, label = parse_filename("img_10.jpg")
    assert is_digit is False
    assert label is None


def test_parse_filename_invalid():
    """Test parsing invalid filenames."""
    is_digit, label = parse_filename("invalid.txt")
    assert is_digit is False
    assert label is None
    
    is_digit, label = parse_filename("no_prefix_3.jpg")
    assert is_digit is False
    assert label is None


def test_parse_filename_case_insensitive():
    """Test that parsing is case-insensitive."""
    is_digit, label = parse_filename("IMG_5.JPG")
    assert is_digit is True
    assert label == 5
