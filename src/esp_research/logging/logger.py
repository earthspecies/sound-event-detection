"""Logging utilities for ESP Research library.

This module provides a convenient logging interface via the `logger` singleton with
sensible defaults. When you call logger methods, a proper logger hierarchy is created
under the hood (e.g., "logger.my_module"), but all loggers ultimately route through this
central logger without propagating to Python's root logger.

This isolation prevents interference with other libraries' logging while maintaining the
benefits of hierarchical logging. Simply import and use:

    from esp_research.logging import logger

    logger.info("Hello from any module!")

The logger automatically determines the calling module's name, so log messages show
their origin without manual configuration.
"""

import inspect
import logging
import os
import warnings
from typing import Any, override

from rich.logging import RichHandler
from rich.traceback import install as use_rich_for_uncaught_exceptions

from .filters import ExceptionOnlyFilter, NoExceptionFilter
from .formatters import ColoredFormatter


class _Formatter(ColoredFormatter):
    """
    A private formatter that removes the "logger." prefix from logger names.

    This provides cleaner log output by stripping the redundant prefix while
    preserving the original record state to avoid side effects. This is a private
    formatter only meant to be used with the logger object exposed in this file.
    """

    @override
    def format(self, record: logging.LogRecord) -> str:
        """
        Wraps format() from ColoredFormatter to remove the common "logger." prefix from
        logger names.

        Returns:
            The formatted log record string.
        """

        name = record.name  # remember
        record.name = record.name.removeprefix("logger.")
        result = super().format(record)
        record.name = name  # reset

        return result


class _Logger:
    """
    A logging singleton that automatically determines the caller's module name.

    This class eliminates the need for the typical `logger =
    logging.getLogger(__name__)` boilerplate in every module. Instead, users import a
    single `logger` instance and use it directly:

        from esp_research.logging import logger
        logger.info("This message automatically shows the current module name")

    The class uses frame inspection to determine which module is calling each logging
    method, then retrieves (or creates) the appropriate child logger for that module.
    All child loggers inherit configuration from the library's root logger via
    propagation.

    The library uses "logger" as its root namespace to avoid interfering with the actual
    root logger that users may configure elsewhere. All library loggers are under this
    namespace (e.g., "logger.esp_research.models.naturelm").
    """

    LOGGER_NAME = "logger"
    FORMAT_STR = "%(levelname)s %(asctime)s [%(name)s] %(message)s"
    DATE_FMT = "%b-%d %-I:%M%p"
    INIT_LEVEL = logging.INFO

    def __init__(self) -> None:
        logger = logging.getLogger(self.LOGGER_NAME)
        logger.setLevel(self.INIT_LEVEL)

        # Stop propagation to avoid interfering with user's root logger
        logger.propagate = False

        # Remove any potentially existing handlers
        logger.handlers.clear()

        formatter = _Formatter(self.FORMAT_STR, datefmt=self.DATE_FMT)

        # Rich can render tracebacks with syntax highlighting and formatting.
        # They're easier to read and show more code than standard Python
        # tracebacks. Let's filter exceptions and send them to RichHandler:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(NoExceptionFilter())
        logger.addHandler(handler)

        exc_handler = RichHandler(rich_tracebacks=True, show_path=False)
        exc_handler.addFilter(ExceptionOnlyFilter())
        logger.addHandler(exc_handler)

        # TODO: Is this the right place for this?
        use_rich_for_uncaught_exceptions()

    def _get_caller_module_name(self) -> str | None:
        """
        Inspect the call stack to determine the calling module's name.

        Walks up the stack frames to find the actual user code that called the logging
        method (skipping internal frames from this class).

        Returns:
            The fully qualified module name (e.g., 'esp_research.models.bert'), or None
            if it cannot be determined from the stack.
        """
        frame = inspect.currentframe()
        try:
            # Go up 3 frames to get to the actual caller
            # _get_caller_module_name -> _get_logger -> logging method -> actual caller
            caller_frame = frame.f_back.f_back.f_back  # pyright: ignore
            caller_module = inspect.getmodule(caller_frame)
            return caller_module.__name__ if caller_module else None
        except AttributeError:
            # This works reliably on CPython, but inspect.currentframe() returns None on
            # Python implementations that don't use stack frames, causing the subsequent
            # .f_back access to fail
            return None
        finally:
            # Frame objects hold references to all local variables in the call stack,
            # which can create reference cycles. Explicitly deleting breaks the cycle
            # and allows prompt garbage collection.
            del frame

    def _get_logger(self) -> logging.Logger:
        """
        Get or create a logger for the current caller's module.

        Returns:
            A logger instance under the 'logger' namespace, named after the caller's
            module (e.g., 'logger.esp_research.models.bert'). This logger inherits
            configuration from the library root logger via propagation.
        """
        caller_name = self._get_caller_module_name()
        if caller_name is None:
            name = self.LOGGER_NAME
        else:
            name = f"{self.LOGGER_NAME}.{caller_name}"

        return logging.getLogger(name)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug-level message."""
        self._get_logger().debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info-level message."""
        self._get_logger().info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning-level message."""
        self._get_logger().warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error-level message."""
        self._get_logger().error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical-level message."""
        self._get_logger().critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        """Log an exception with traceback. Typically called from an exception handler."""
        self._get_logger().exception(msg, *args, exc_info=exc_info, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a message at the specified level."""
        self._get_logger().log(level, msg, *args, **kwargs)

    def setLevel(self, level: int) -> None:
        """
        Change the logging level for all loggers under `logger`

        Args:
            level: The new logging level (e.g., logging.DEBUG, logging.INFO)

        Example:
            >>> from esp_research.logging import logger
            >>> import logging
            >>> logger.setLevel(logging.DEBUG)  # Enable debug messages
        """
        logger = logging.getLogger(self.LOGGER_NAME)
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    def addHandler(self, handler: logging.Handler, this_module_only: bool = False) -> None:
        """
        Add a handler to the logger hierarchy.

        Args:
            handler: The logging handler to add
            this_module_only: If True, adds the handler only to the calling
                module's logger. If False, adds the handler to the root logger,
                affecting all modules.

        Example:
            >>> from esp_research.logging import logger
            >>> import logging
            >>> # Add a file handler for all modules
            >>> file_handler = logging.FileHandler('app.log')
            >>> logger.addHandler(file_handler)
            >>>
            >>> # Add a handler only for this specific module
            >>> debug_handler = logging.FileHandler('this_module_debug.log')
            >>> logger.addHandler(debug_handler, this_module_only=True)
        """
        if this_module_only:
            logger = self._get_logger()
        else:
            logger = logging.getLogger(self.LOGGER_NAME)

        logger.addHandler(handler)


# Singleton logger instance - import and use this directly in your code
logger = _Logger()


# Suppress TensorFlow C++ logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error

# Filter the specific warning from tf2onnx_lib.py
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.src.export.tf2onnx_lib")
