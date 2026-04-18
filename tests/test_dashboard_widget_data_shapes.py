"""
Contract tests: widget ``get_data`` output must contain the field names the
frontend renderers read.

The dashboard widgets have suffered from silent backend⇄frontend drift
(e.g. backend ``wind_speed`` vs frontend ``wind``, backend
``active_count`` vs frontend ``count``). This file locks the backend
shape in place — if a widget removes or renames a field, these tests
break loudly and point at the file the frontend expects.

We check **backend keys only**, not their runtime values (which depend
on live services). The frontend code is the source of truth for what's
needed; we mirror that list here.
"""

from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.dashboard.widgets.memory_stats_widget import MemoryStatsWidget
from plugins.dashboard.widgets.sessions_widget import SessionsWidget
from plugins.dashboard.widgets.system_status_widget import SystemStatusWidget
from plugins.dashboard.widgets.thoughts_widget import ThoughtsWidget


def _api(**overrides: Any) -> MagicMock:
    api = MagicMock()
    api._app = MagicMock()
    api._app.memory = overrides.get("memory", None)
    api._app.session_store = overrides.get("session_store", None)
    api._app.plugin_loader = overrides.get("plugin_loader", None)
    api.get_plugin.return_value = overrides.get("plugin", None)
    return api


# ─── Memory Stats ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_stats_shape_when_unavailable() -> None:
    w = MemoryStatsWidget(_api(memory=None))
    data = await w.get_data()
    # Frontend reads: available, collections, total, fts_count
    for key in ("available", "collections", "total", "fts_count"):
        assert key in data, f"memory_stats missing {key!r}"
    assert data["available"] is False
    assert data["collections"] == {}


@pytest.mark.asyncio
async def test_memory_stats_shape_when_available() -> None:
    memory = MagicMock()
    # Two collections with count()
    col1 = MagicMock()
    col1.count.return_value = 7
    col2 = MagicMock()
    col2.count.return_value = 3
    memory._collections = {"facts": col1, "context": col2}
    memory._fts = AsyncMock()
    memory._fts.execute = AsyncMock(
        return_value=MagicMock(
            fetchone=AsyncMock(return_value=(42,)),
            close=AsyncMock(),
        )
    )

    w = MemoryStatsWidget(_api(memory=memory))
    data = await w.get_data()
    assert data["available"] is True
    assert data["collections"] == {"facts": 7, "context": 3}
    assert data["total"] == 10
    assert data["fts_count"] == 42


# ─── Sessions ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sessions_shape_contains_frontend_fields() -> None:
    store = MagicMock()
    store.sessions.return_value = ["sess-1", "sess-2"]
    store.get.side_effect = lambda sid: (
        [{"role": "user", "content": "hi"}] if sid == "sess-1"
        else [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    )
    w = SessionsWidget(_api(session_store=store))
    data = await w.get_data()

    # Frontend reads: active_count, sessions[] (with id/messages/last_snippet),
    # total_messages
    assert "active_count" in data
    assert "sessions" in data
    assert "total_messages" in data
    assert data["active_count"] == 2
    assert data["total_messages"] == 3
    assert isinstance(data["sessions"], list)
    for s in data["sessions"]:
        assert "id" in s
        assert "messages" in s
        assert "last_snippet" in s


# ─── Thoughts ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thoughts_shape_contains_frontend_fields() -> None:
    api = _api()
    w = ThoughtsWidget(api)
    w._on_thought({"mode": "daydream", "text": "a"})
    data = await w.get_data()
    # Frontend reads: thoughts[], count, enabled. Each thought has mode/text/at.
    assert "thoughts" in data
    assert "count" in data
    assert "enabled" in data
    assert data["count"] == 1
    t = data["thoughts"][0]
    for key in ("mode", "text", "at"):
        assert key in t, f"thought entry missing {key!r}"


# ─── System Status ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_system_status_shape_contains_frontend_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    api._app.plugin_loader = MagicMock(loaded_count=5)
    w = SystemStatusWidget(api)

    # Stub out httpx so the widget doesn't actually try to hit anything.
    import httpx

    class _FakeClient:
        def __init__(self, *_, **__) -> None: ...
        async def __aenter__(self) -> "_FakeClient": return self
        async def __aexit__(self, *_exc: Any) -> None: return None
        async def get(self, *_args: Any, **_kwargs: Any) -> Any:
            raise httpx.ConnectError("stubbed")

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    data = await w.get_data()

    # Frontend reads: services[], plugins_loaded, uptime_seconds.
    # Each service has name, host, status.
    assert "services" in data
    assert "plugins_loaded" in data
    assert "uptime_seconds" in data
    assert data["plugins_loaded"] == 5
    assert isinstance(data["services"], list)
    for svc in data["services"]:
        assert "name" in svc
        assert "host" in svc
        assert "status" in svc, (
            "svc must have 'status' field (frontend reads it as string: "
            "up/down/timeout/error)"
        )
