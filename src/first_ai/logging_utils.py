import logging
from pathlib import Path
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    to_file: Optional[Path] = None,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """
    Create and configure a logger with stream (stdout) and optional file handler.

    Args:
        name: Optional logger name. Defaults to root when None.
        level: Logging level, e.g., logging.INFO.
        to_file: Optional path to write logs.
        fmt: Log message format.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if configure_logger is called multiple times
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(fmt))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

        if to_file is not None:
            to_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(to_file)
            file_handler.setFormatter(logging.Formatter(fmt))
            file_handler.setLevel(level)
            logger.addHandler(file_handler)

    return logger
