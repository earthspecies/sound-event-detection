"""Convenient functions for configuring logging."""

import logging

from .formatters import ColoredFormatter


def setup_logging(
    level: int = logging.INFO,
    format_str: str = "%(levelname)s %(asctime)s [%(name)s] %(message)s",
    datefmt: str = "%b-%d %-I:%M%p",
) -> None:
    """
    Configure Python's root logger with colored console output.

    This is a convenience function for quickly setting up logging for scripts and
    applications. It configures the root logger with a StreamHandler using the
    ColoredFormatter.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO). Defaults to INFO.
        format_str: Log message format string. Defaults to a format showing level,
            timestamp, logger name, and message.
        datefmt: Date/time format string. Defaults to "Mon-25 3:45PM" style.

    Example:
        >>> from esp_research.logging import setup_logging
        >>> import logging
        >>> setup_logging()
        >>> logging.info("Hello, world!")
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(format_str, datefmt=datefmt))

    root_logger.addHandler(handler)
