"""Logging filters for ESP Research library.

This module provides filters that control which log records are processed by
handlers. Filters can be attached to handlers to selectively process or ignore
certain types of log records, such as those with or without exception
information.
"""

import logging
from typing import override


class NoExceptionFilter(logging.Filter):
    """
    A filter that excludes log records containing exception information.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.exc_info is None


class ExceptionOnlyFilter(logging.Filter):
    """
    A filter that only accepts log records containing exception information.

    This allows a dedicated handler to process only exceptions while ignoring
    regular log messages.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        return record.exc_info is not None
