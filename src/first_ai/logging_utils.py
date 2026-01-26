"""Shared logging configuration helpers.

Used by the first_ai CLI and can be imported by training/detection scripts
to ensure consistent logging outputs and optional file handlers.
"""

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:  # Defer torch import so modules without torch still import helpers safely
    import torch
except Exception:  # pragma: no cover - runtime guard
    torch = None  # type: ignore

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


def get_environment_info() -> Dict[str, Any]:
    """Collect environment details for consistent header logging."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    env_name = "base"
    if os.environ.get("VIRTUAL_ENV"):
        env_name = f"venv ({os.path.basename(os.environ['VIRTUAL_ENV'])})"
    elif os.environ.get("CONDA_DEFAULT_ENV"):
        env_name = os.environ["CONDA_DEFAULT_ENV"]

    cuda_available = torch.cuda.is_available() if torch else False
    cuda_version = torch.version.cuda if (torch and cuda_available) else None
    gpu_name = torch.cuda.get_device_name(0) if (torch and cuda_available) else None

    cpu_name = None
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:
                cpu_name = lines[1].strip()
        except Exception:
            cpu_name = None

    if not cpu_name:
        cpu_name = platform.processor() or f"Unknown CPU ({platform.machine()})"

    return {
        "env_name": env_name,
        "python_version": python_version,
        "pytorch_version": torch.__version__ if torch else "unknown",
        "cuda_version": cuda_version,
        "cpu_name": cpu_name,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def log_environment_block(logger: logging.Logger, env_info: Optional[Dict[str, Any]] = None) -> None:
    """Log a standardized environment header (aligned with the dashboard)."""
    env = env_info or get_environment_info()

    logger.info("\n" + "=" * 80)
    logger.info("  Environment".center(80))
    logger.info("=" * 80)
    logger.info(f"  Environment: {env['env_name']}")

    python_line = f"  Python: {env['python_version']} | PyTorch: {env['pytorch_version']}"
    if env.get("cuda_version"):
        python_line += f" | CUDA: {env['cuda_version']}"
    logger.info(python_line)

    logger.info(f"  CPU: {env['cpu_name']}")
    if env.get("cuda_available"):
        logger.info(f"  GPU: {env['gpu_name']}")
        logger.info(f"  ⚡ Compute Device: GPU - {env['gpu_name']}")
    else:
        logger.info("  GPU: None detected")
        logger.info("  💻 Compute Device: CPU")
    logger.info("=" * 80)
