"""Lexy AI – FastAPI Gateway + WebSocket."""

from lexy_core.websocket.server import WSServer
from lexy_core.websocket.gateway import build_app

__all__ = ["WSServer", "build_app"]
