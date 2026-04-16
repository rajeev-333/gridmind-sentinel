"""
Structured Logging Module — src/utils/logger.py

Provides a pre-configured logger with structured formatting for all
GridMind Sentinel modules. Uses Python's built-in logging with a custom
formatter that includes timestamp, module name, level, and message.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("RAG pipeline initialized", extra={"chunks": 42})

Connection to system:
    - Every module imports get_logger() for consistent log formatting.
    - Log level is controlled via settings.LOG_LEVEL (from .env).
"""

import logging
import sys
from src.utils.config import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured, readable log lines."""

    FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for the given module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger
