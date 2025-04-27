"""
Logging configuration for the OCR POC project.

This module sets up a global logger with appropriate handlers and formatters.
It follows the philosophy of configuring the root logger without modifying
individual loggers.
"""

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from ..config.settings import settings


def setup_logger(
    log_file: Optional[Union[str, Path]] = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Set up the global logger configuration.

    Args:
        log_file: Path to the log file. If None, logs will only go to stderr.
        log_level: Minimum log level to display.
        rotation: When to rotate the log file.
        retention: How long to keep the log files.
    """
    # Remove default logger
    logger.remove()

    # Add stderr handler with custom format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.debug("Logger initialized with level: {}", log_level)


# Initialize logger
setup_logger(
    log_file=Path(settings.project_dir) / "logs" / "ocr_poc.log" if not settings.debug else None,
    log_level="DEBUG" if settings.debug else "INFO",
)


__all__ = ["logger", "setup_logger"]
