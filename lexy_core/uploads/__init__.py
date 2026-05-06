"""
Lexy AI — File-Upload subsystem.

Top-level entry point: :class:`UploadHandler`. Receives a FastAPI
``UploadFile`` plus a session id and routes the bytes through the right
processor (image / document / code / audio), then returns a manifest
dict the frontend can hand back when sending the chat message.

The handler is intentionally a single class with one async ``handle()``
method per kind. That keeps the gateway thin (one route per kind, each
just calls ``handler.handle_image(...)`` etc.) and keeps the kind-specific
logic in :mod:`lexy_core.uploads.processors` where it can be unit-tested
without spinning up FastAPI.
"""

from __future__ import annotations

from .handler import UploadHandler, UploadResult, UploadKind
from .store import UploadStore, UploadRecord

__all__ = [
    "UploadHandler",
    "UploadResult",
    "UploadKind",
    "UploadStore",
    "UploadRecord",
]
