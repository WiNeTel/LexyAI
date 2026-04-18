"""
Tests for the multi-path service probe in SystemStatusWidget.

The widget must:
* try each candidate path in order
* report 'up' on first 2xx/3xx response
* fall through to the next path on 4xx/5xx (e.g. ChromaDB 1.0 returns
  410 Gone on the legacy ``/api/v1/heartbeat``)
* short-circuit to 'down' on ConnectError (no listener at all)
* report 'timeout' if every candidate path times out
* remember which path succeeded so the frontend can display it
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from plugins.dashboard.widgets.system_status_widget import SystemStatusWidget


def _api() -> MagicMock:
    api = MagicMock()
    api._app = MagicMock()
    api._app.plugin_loader = None
    return api


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeClient:
    """Async client stub: maps URL → response/exception."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.calls.append(url)
        reply = self._responses.get(url)
        if reply is None:
            raise httpx.ConnectError(f"no stub for {url}")
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.mark.asyncio
async def test_first_path_up_marks_service_up() -> None:
    client = _FakeClient({
        "http://127.0.0.1:8000/api/v2/heartbeat": _FakeResponse(200),
    })
    svc = {
        "name": "ChromaDB",
        "host": "127.0.0.1",
        "port": "8000",
        "paths": ["/api/v2/heartbeat", "/api/v1/heartbeat"],
    }
    widget = SystemStatusWidget(_api())
    entry = await widget._probe_service(client, svc)  # type: ignore[arg-type]
    assert entry["status"] == "up"
    assert entry["path"] == "/api/v2/heartbeat"
    # Short-circuit: only the first path was tried
    assert client.calls == ["http://127.0.0.1:8000/api/v2/heartbeat"]


@pytest.mark.asyncio
async def test_falls_through_from_4xx_to_next_path() -> None:
    """ChromaDB 1.0 returns 410 on v1; widget must then try v2."""
    client = _FakeClient({
        "http://127.0.0.1:8000/api/v1/heartbeat": _FakeResponse(410),
        "http://127.0.0.1:8000/api/v2/heartbeat": _FakeResponse(200),
    })
    svc = {
        "name": "ChromaDB",
        "host": "127.0.0.1",
        "port": "8000",
        "paths": ["/api/v1/heartbeat", "/api/v2/heartbeat"],
    }
    widget = SystemStatusWidget(_api())
    entry = await widget._probe_service(client, svc)  # type: ignore[arg-type]
    assert entry["status"] == "up"
    assert entry["path"] == "/api/v2/heartbeat"
    # Both URLs were tried
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_all_paths_return_4xx_keeps_error_with_status_code() -> None:
    client = _FakeClient({
        "http://127.0.0.1:8000/api/v1/heartbeat": _FakeResponse(410),
        "http://127.0.0.1:8000/api/v2/heartbeat": _FakeResponse(404),
    })
    svc = {
        "name": "ChromaDB",
        "host": "127.0.0.1",
        "port": "8000",
        "paths": ["/api/v1/heartbeat", "/api/v2/heartbeat"],
    }
    widget = SystemStatusWidget(_api())
    entry = await widget._probe_service(client, svc)  # type: ignore[arg-type]
    assert entry["status"] == "error"
    assert entry["status_code"] == 404  # the LAST path's code
    assert entry["path"] == "/api/v2/heartbeat"


@pytest.mark.asyncio
async def test_connect_error_short_circuits_to_down() -> None:
    """If nothing is listening, don't waste time trying other paths."""
    client = _FakeClient({})  # every URL → ConnectError
    svc = {
        "name": "ChromaDB",
        "host": "127.0.0.1",
        "port": "8000",
        "paths": ["/api/v2/heartbeat", "/api/v1/heartbeat", "/"],
    }
    widget = SystemStatusWidget(_api())
    entry = await widget._probe_service(client, svc)  # type: ignore[arg-type]
    assert entry["status"] == "down"
    # Short-circuited after the first ConnectError
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_all_timeouts_report_timeout_status() -> None:
    client = _FakeClient({
        "http://127.0.0.1:8000/api/v2/heartbeat": httpx.TimeoutException("slow"),
        "http://127.0.0.1:8000/api/v1/heartbeat": httpx.TimeoutException("slow"),
    })
    svc = {
        "name": "ChromaDB",
        "host": "127.0.0.1",
        "port": "8000",
        "paths": ["/api/v2/heartbeat", "/api/v1/heartbeat"],
    }
    widget = SystemStatusWidget(_api())
    entry = await widget._probe_service(client, svc)  # type: ignore[arg-type]
    assert entry["status"] == "timeout"
    # Both candidates were tried (timeout is recoverable, ConnectError isn't)
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_legacy_single_path_string_still_works() -> None:
    """Backward-compat: services using the old 'path' (str) shape keep working."""
    client = _FakeClient({
        "http://127.0.0.1:5005/health": _FakeResponse(200),
    })
    svc = {
        "name": "E4B Brain",
        "host": "127.0.0.1",
        "port": "5005",
        "path": "/health",  # legacy key, string value
    }
    widget = SystemStatusWidget(_api())
    entry = await widget._probe_service(client, svc)  # type: ignore[arg-type]
    assert entry["status"] == "up"
    assert entry["path"] == "/health"


@pytest.mark.asyncio
async def test_chromadb_default_config_uses_v2_first() -> None:
    """Sanity check: the module's default _SERVICES tries v2 before v1."""
    from plugins.dashboard.widgets.system_status_widget import _SERVICES

    chroma = next(s for s in _SERVICES if s["name"] == "ChromaDB")
    assert chroma["paths"][0] == "/api/v2/heartbeat", (
        "ChromaDB probe must try v2 first — v1 returns 410 Gone on "
        "ChromaDB 1.0+"
    )
    assert "/api/v1/heartbeat" in chroma["paths"], (
        "Legacy v1 should still be in the fallback list for older ChromaDB "
        "installations"
    )
