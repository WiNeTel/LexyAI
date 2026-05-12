"""
Lexy AI - Structured Logging (structlog).

Single point of configuration. Every module uses
    from lexy_core.utils import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_configured = False


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """
    Configure structlog + stdlib logging once.

    Args:
        level: Log level name (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        json_output: If True, emit JSON logs (production); otherwise console renderer.
    """
    global _configured

    log_level = _LEVEL_MAP.get(level.upper(), logging.INFO)

    # Stdlib root: route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Silence noisy library loggers. They emit one INFO line per HTTP
    # request — and we have channel-poll loops + chromadb count probes
    # firing every 1-2 s, which buries Lexy's own structlog output.
    # Errors still surface (WARNING/ERROR pass through).
    for _noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None, **bound: Any) -> structlog.BoundLogger:
    """
    Get a structlog logger. Lazy-configures with INFO if not configured yet.
    """
    if not _configured:
        configure_logging()
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    if bound:
        logger = logger.bind(**bound)
    return logger
