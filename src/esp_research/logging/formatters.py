"""Convenient logging formatters. Formatters are responsible for _how_ messages are
formatted"""

import json
import logging
from datetime import datetime, timezone
from typing import Any


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds ANSI color codes to console output.

    This formatter enhances log records with color-coded level names and timestamps for
    improved readability in terminal output. It maps log levels to distinct colors.
    Level names are also abbreviated (e.g., "ERROR" -> "ERR") and displayed in bold.

    The formatter uses ANSI escape codes and is designed for console/terminal output.
    Timestamps are displayed in grey for visual separation.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREY = "\033[90m"
    WHITE = "\033[97m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"

    _COLOR_MAP = {
        "DEBUG": WHITE,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": BRIGHT_RED,
    }

    _ABBREV_MAP = {
        "CRITICAL": "CRT",
        "ERROR": "ERR",
        "WARNING": "WAR",
        "INFO": "INF",
        "DEBUG": "DBG",
        "NOTSET": "NST",
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.

        Returns:
            Formatted log string with ANSI color codes.
        """

        # remember the original levelname
        levelname = record.levelname

        if levelname in self._COLOR_MAP:
            level_abbrev = self._ABBREV_MAP.get(levelname, levelname[:3].upper())
            record.levelname = f"{self.BOLD}{self._COLOR_MAP[levelname]}{level_abbrev}{self.RESET}"

        result = super().format(record)

        # Reset levelname to avoid side effects
        record.levelname = levelname

        return result

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """
        Format the time with grey color.

        Returns:
            Time string with grey ANSI color codes.
        """
        time_str = super().formatTime(record, datefmt)
        return f"{self.GREY}{time_str}{self.RESET}"


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        return json.dumps(log_data)
