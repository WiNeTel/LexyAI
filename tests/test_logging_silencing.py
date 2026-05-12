"""
Pin that ``configure_logging`` mutes the noisy HTTP-client library
loggers.

Background: ``httpx`` and ``chromadb`` log every single request at
INFO level. With the WhatsApp/Telegram bridge polling
``http://127.0.0.1:3000/inbound`` every 2 s and the dashboard
hitting ChromaDB count probes once per second, the console gets
buried in ``HTTP Request: GET ... "HTTP/1.1 200 OK"`` lines and
Lexy's own structlog output is unreadable. We push these library
loggers to WARNING so only real errors surface.
"""

from __future__ import annotations

import logging

from lexy_core.utils.logging import configure_logging


def test_httpx_logger_silenced_after_configure() -> None:
    configure_logging(level="INFO")
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_chromadb_logger_silenced_after_configure() -> None:
    configure_logging(level="INFO")
    assert logging.getLogger("chromadb").level >= logging.WARNING


def test_httpcore_and_urllib3_silenced() -> None:
    configure_logging(level="INFO")
    assert logging.getLogger("httpcore").level >= logging.WARNING
    assert logging.getLogger("urllib3").level >= logging.WARNING


def test_silencing_survives_reconfigure_with_debug() -> None:
    """Even when the operator switches the root level to DEBUG to
    diagnose something, the noisy libs stay at WARNING — otherwise
    a DEBUG run drowns in poll-loop noise before any useful
    ``agent.plan`` line appears."""
    configure_logging(level="DEBUG")
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("chromadb").level >= logging.WARNING
