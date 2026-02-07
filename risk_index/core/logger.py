"""Structured logging setup for the risk index system."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from risk_index.core.constants import LOGS_DIR


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            ):
                log_data[key] = value

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} {record.getMessage()}"

        # Add context fields if present
        extras = []
        for key in ("ticker", "source", "series", "block", "step"):
            if hasattr(record, key):
                extras.append(f"{key}={getattr(record, key)}")
        if extras:
            message += f" ({', '.join(extras)})"

        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


def setup_logger(
    name: str = "risk_index",
    level: int = logging.INFO,
    log_file: bool = True,
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with JSON file and console handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Whether to log to file
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ConsoleFormatter())
        logger.addHandler(console_handler)

    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        log_path = LOGS_DIR / f"risk_index_{date_str}.jsonl"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "risk_index") -> logging.Logger:
    """Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LogContext:
    """Context manager for adding context to log messages."""

    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        context = self.context

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args):
        logging.setLogRecordFactory(self.old_factory)


def log_step(logger: logging.Logger, step: str, status: str = "start", **extra: Any) -> None:
    """Log a pipeline step.

    Args:
        logger: Logger instance
        step: Step name
        status: Step status ('start', 'complete', 'error')
        **extra: Additional context
    """
    extra["step"] = step
    extra["status"] = status

    if status == "start":
        logger.info(f"Starting {step}", extra=extra)
    elif status == "complete":
        logger.info(f"Completed {step}", extra=extra)
    elif status == "error":
        logger.error(f"Error in {step}", extra=extra)
    else:
        logger.info(f"{step}: {status}", extra=extra)


def log_data_quality(
    logger: logging.Logger,
    series_id: str,
    start_date: str,
    end_date: str,
    null_pct: float,
    status: str,
) -> None:
    """Log data quality metrics.

    Args:
        logger: Logger instance
        series_id: Series identifier
        start_date: Data start date
        end_date: Data end date
        null_pct: Percentage of null values
        status: Quality status ('current', 'delayed', 'discontinued')
    """
    logger.info(
        f"Data quality: {series_id}",
        extra={
            "series": series_id,
            "start_date": start_date,
            "end_date": end_date,
            "null_pct": round(null_pct, 4),
            "status": status,
        },
    )
