"""
Tests for push-based thoughts widget updates.

When the autonomous_thinking plugin emits ``core.autonomous_thought``:
1. The ThoughtsWidget's own handler appends the entry to its cache.
2. The DashboardPlugin's handler broadcasts the widget's current data.

Subscription order (widget first, then dashboard) is important — if the
dashboard fires before the widget caches, the broadcast would miss the
new thought. These tests verify:

* ThoughtsWidget._on_thought accepts both dict payloads and Event objects.
* The dashboard handler broadcasts after the widget cache updates.
* Unknown event shape does not crash the handler.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.dashboard.dashboard_plugin import DashboardPlugin
from plugins.dashboard.widgets.thoughts_widget import ThoughtsWidget


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_api() -> MagicMock:
    api = MagicMock()
    api.get_config.return_value = {}
    api.ws_broadcast = AsyncMock()
    api.on_event = MagicMock()
    api.get_plugin.return_value = None  # No thinking plugin → enabled=False
    api._app = MagicMock()
    api._app.memory = None
    return api


# ─── ThoughtsWidget._on_thought robustness ────────────────────────────────


class TestThoughtsWidgetOnThought:
    def test_handles_dict_payload(self) -> None:
        api = _make_api()
        w = ThoughtsWidget(api)
        w._on_thought({"mode": "daydream", "text": "test thought"})
        assert len(w._cache) == 1
        assert w._cache[0]["mode"] == "daydream"
        assert w._cache[0]["text"] == "test thought"

    def test_handles_event_object(self) -> None:
        """The EventBus delivers an Event object, not a dict."""
        api = _make_api()
        w = ThoughtsWidget(api)
        event = SimpleNamespace(
            name="core.autonomous_thought",
            data={"mode": "reflect", "text": "another thought"},
            source="autonomous_thinking",
        )
        w._on_thought(event)
        assert len(w._cache) == 1
        assert w._cache[0]["mode"] == "reflect"
        assert w._cache[0]["text"] == "another thought"

    def test_missing_fields_use_defaults(self) -> None:
        api = _make_api()
        w = ThoughtsWidget(api)
        w._on_thought({})
        assert w._cache[0]["mode"] == "unknown"
        assert w._cache[0]["text"] == ""
        assert w._cache[0]["at"]  # HH:MM string

    def test_non_dict_non_event_does_not_crash(self) -> None:
        api = _make_api()
        w = ThoughtsWidget(api)
        # Passing a bare string should not raise.
        w._on_thought("not a dict or event")  # type: ignore[arg-type]
        assert len(w._cache) == 1
        assert w._cache[0]["mode"] == "unknown"


# ─── get_data shape ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_data_returns_thoughts_newest_first() -> None:
    api = _make_api()
    w = ThoughtsWidget(api)
    w._on_thought({"mode": "a", "text": "first"})
    w._on_thought({"mode": "b", "text": "second"})
    w._on_thought({"mode": "c", "text": "third"})
    data = await w.get_data()
    assert data["count"] == 3
    # Newest first
    assert data["thoughts"][0]["text"] == "third"
    assert data["thoughts"][-1]["text"] == "first"


@pytest.mark.asyncio
async def test_get_data_reports_disabled_when_plugin_missing() -> None:
    api = _make_api()
    api.get_plugin.return_value = None
    w = ThoughtsWidget(api)
    data = await w.get_data()
    assert data["enabled"] is False


@pytest.mark.asyncio
async def test_get_data_reports_enabled_flag_from_plugin() -> None:
    api = _make_api()
    thinking_plugin = MagicMock(spec=["_thinking_active"])
    thinking_plugin._thinking_active = True
    api.get_plugin.return_value = thinking_plugin
    w = ThoughtsWidget(api)
    data = await w.get_data()
    assert data["enabled"] is True


# ─── DashboardPlugin broadcast on new thought ─────────────────────────────


@pytest.mark.asyncio
async def test_on_autonomous_thought_broadcasts_widget_data() -> None:
    """End-to-end: event → widget cache updated → dashboard broadcasts."""
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    # Install a real ThoughtsWidget into the plugin's instance map so the
    # dashboard can broadcast its data via get_data().
    widget = ThoughtsWidget(api)
    plugin._widget_instances["thoughts"] = widget

    async def data_fn() -> dict[str, Any]:
        return await widget.get_data()

    plugin.register_widget(
        widget_id="thoughts",
        data_fn=data_fn,
        refresh_interval=0.0,  # push-only
        default_size=(3, 2),
        title="Gedanken",
        source="dashboard",
    )

    # Simulate the subscription order that on_enable sets up:
    # widget first (populates cache), dashboard second (broadcasts).
    widget._on_thought({"mode": "daydream", "text": "a fresh thought"})
    await plugin._on_autonomous_thought(
        SimpleNamespace(name="core.autonomous_thought", data={})
    )

    # One broadcast with the just-cached thought
    api.ws_broadcast.assert_awaited_once()
    msg = api.ws_broadcast.await_args.args[0]
    assert msg["type"] == "dashboard_widget_update"
    assert msg["widget_id"] == "thoughts"
    assert msg["data"]["count"] == 1
    assert msg["data"]["thoughts"][0]["text"] == "a fresh thought"


@pytest.mark.asyncio
async def test_on_autonomous_thought_without_widget_does_not_crash() -> None:
    """If thoughts widget isn't registered, handler silently does nothing."""
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())
    # No widget registered
    await plugin._on_autonomous_thought(
        SimpleNamespace(name="core.autonomous_thought", data={})
    )
    api.ws_broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_autonomous_thought_swallows_broadcast_errors() -> None:
    api = _make_api()
    api.ws_broadcast = AsyncMock(side_effect=RuntimeError("ws dead"))
    plugin = DashboardPlugin(api, MagicMock())

    widget = ThoughtsWidget(api)
    plugin._widget_instances["thoughts"] = widget

    async def data_fn() -> dict[str, Any]:
        return await widget.get_data()

    plugin.register_widget(
        widget_id="thoughts",
        data_fn=data_fn,
        refresh_interval=0.0,
        default_size=(3, 2),
        title="X",
        source="d",
    )

    # Must not raise even though ws_broadcast throws
    await plugin._on_autonomous_thought(
        SimpleNamespace(name="core.autonomous_thought", data={})
    )
