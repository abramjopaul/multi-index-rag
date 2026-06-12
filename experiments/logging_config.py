"""
Global logging configuration utility for experiment and test scripts.

Usage in your test/experiment scripts:
    from logging_config import configure_logging
    configure_logging()  # Uses CLI argument or defaults to INFO

    # Or explicitly:
    configure_logging(level="DEBUG")
"""

import logging
import sys
from typing import Literal, Optional


def configure_logging(
    level: Optional[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]] = None,
) -> None:
    """
    Configure logging globally with CLI support.

    Accepts log level from:
    1. Explicit argument to this function
    2. First CLI argument (sys.argv[1])
    3. Default: INFO

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               If None, reads from CLI argument or defaults to INFO

    Example:
        >>> configure_logging()  # CLI arg or INFO
        >>> configure_logging(level="DEBUG")  # Explicit DEBUG
    """
    # Priority: function arg > CLI arg > default
    if level is None:
        level = sys.argv[1].upper() if len(sys.argv) > 1 else "INFO" #type: ignore
    else:
        level = level.upper() #type: ignore

    # Validate and get the logging level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in valid_levels:
        print(f"Invalid log level: {level}. Valid options: {', '.join(valid_levels)}")
        print("Using INFO instead.\n")
        level_int = logging.INFO
    else:
        level_int = getattr(logging, level)

    # Configure root logger
    logging.basicConfig(
        level=level_int, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured at level: {level}")
