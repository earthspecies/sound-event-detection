from .formatters import ColoredFormatter, JSONFormatter
from .handlers import get_gcp_handler
from .logger import logger
from .logging import setup_logging

__all__ = [
    "ColoredFormatter",
    "JSONFormatter",
    "setup_logging",
    "get_gcp_handler",
    "logger",
]

# TODO: Add unit tests once we're happy with the design
# TODO: add unit tests for custom logging levels
