"""Unit tests for src.first_ai.logging_utils module."""

import logging
from pathlib import Path
import tempfile
import pytest

from src.first_ai.logging_utils import configure_logger


def test_configure_logger_default():
    """Test logger creation with default settings."""
    logger = configure_logger("test_logger", level=logging.INFO)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_configure_logger_with_file():
    """Test logger creation with file handler."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "logs" / "test.log"
        logger = configure_logger("test_file_logger", level=logging.DEBUG, to_file=log_file)
        
        assert log_file.exists()
        logger.info("Test message")
        
        content = log_file.read_text()
        assert "Test message" in content
        
        # Close handlers to release file lock on Windows
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def test_configure_logger_no_duplicate_handlers():
    """Test that calling configure_logger multiple times doesn't duplicate handlers."""
    logger = configure_logger("test_dup", level=logging.INFO)
    handler_count1 = len(logger.handlers)
    
    # Call again
    logger = configure_logger("test_dup", level=logging.INFO)
    handler_count2 = len(logger.handlers)
    
    assert handler_count1 == handler_count2, "Should not add duplicate handlers"


def test_configure_logger_custom_format():
    """Test logger with custom format."""
    custom_fmt = "%(levelname)s: %(message)s"
    logger = configure_logger("test_fmt", level=logging.INFO, fmt=custom_fmt)
    
    assert logger.level == logging.INFO
    # Verify format by checking handler formatter
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            assert handler.formatter is not None
