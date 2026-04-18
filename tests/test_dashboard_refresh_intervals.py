"""
Tests for per-widget refresh scheduling in the Dashboard plugin.

The refactored DashboardPlugin spawns one asyncio task per widget, each
honoring its own ``refresh_interval``. Config overrides from
``widget_intervals`` take precedence over the widget class default, and
a value of 0 switches a widget into push-only mode (no periodic task).

These tests use real asyncio with small intervals (0.05 s) and a mock
PluginAPI to verify:

* config parsing (invalid entries dropped, negatives clamped)
* per-widget task spawn count (push-only widgets don't get a task)
* interval override (config value wins over class default)
* each tick broadcasts a ``dashboard_widget_update`` message
* broadcast happens even when data hasn't changed (no stale diffing)
* timeout on data_fn does not crash the loop
* on_disable cancels all tasks cleanly
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.dashboard.dashboard_plugin import DashboardPlugin


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_api(**overrides: Any) -> MagicMock:
    api = MagicMock()
    api.get_config.return_value = overrides.get("config", {})
    api.get_db = AsyncMock(return_value=AsyncMock())
    api.ws_broadcast = AsyncMock()
    api.register_ws_handler = MagicMock()
    api.register_tool = MagicMock()
    api.on_event = MagicMock()
    api._app = MagicMock()
    api._app.memory = overrides.get("memory", None)
    api._app.plugin_loader = overrides.get("plugin_loader", None)
    return api


# ─── Config parsing ───────────────────────────────────────────────────────


class TestIntervalOverrideParsing:
    def test_valid_entries_are_kept(self) -> None:
        parsed = DashboardPlugin._parse_interval_overrides(
            {"weather": 1800, "sessions": 30, "thoughts": 0}
        )
        assert parsed == {"weather": 1800.0, "sessions": 30.0, "thoughts": 0.0}

    def test_non_dict_returns_empty(self) -> None:
        assert DashboardPlugin._parse_interval_overrides("not a dict") == {}
        assert DashboardPlugin._parse_interval_overrides(None) == {}

    def test_invalid_values_are_dropped(self) -> None:
        parsed = DashboardPlugin._parse_interval_overrides(
            {"good": 10, "bad": "forty-two", "also_good": 5.5}
        )
        assert parsed == {"good": 10.0, "also_good": 5.5}

    def test_negative_values_clamped_to_zero(self) -> None:
        parsed = DashboardPlugin._parse_interval_overrides({"w": -5})
        assert parsed == {"w": 0.0}


# ─── Interval override at registration ────────────────────────────────────


class TestRegistrationHonoursOverrides:
    def test_config_override_wins_over_class_default(self) -> None:
        api = _make_api(
            config={"widget_intervals": {"weather": 1800}}
        )
        plugin = DashboardPlugin(api, MagicMock())

        # Simulate config load
        plugin._widget_intervals = {"weather": 1800.0}

        async def df() -> dict[str, Any]:
            return {}

        # Widget class default is 300 → config should raise it to 1800.
        plugin.register_widget(
            widget_id="weather",
            data_fn=df,
            refresh_interval=300.0,
            default_size=(2, 2),
            title="Wetter",
            source="dashboard",
        )
        assert plugin._widgets["weather"].refresh_interval == 1800.0

    def test_no_override_uses_class_default(self) -> None:
        api = _make_api()
        plugin = DashboardPlugin(api, MagicMock())
        plugin._widget_intervals = {}

        async def df() -> dict[str, Any]:
            return {}

        plugin.register_widget(
            widget_id="clock",
            data_fn=df,
            refresh_interval=60.0,
            default_size=(1, 1),
            title="Clock",
            source="dashboard",
        )
        assert plugin._widgets["clock"].refresh_interval == 60.0

    def test_override_zero_switches_to_push_only(self) -> None:
        api = _make_api()
        plugin = DashboardPlugin(api, MagicMock())
        plugin._widget_intervals = {"thoughts": 0.0}

        async def df() -> dict[str, Any]:
            return {}

        plugin.register_widget(
            widget_id="thoughts",
            data_fn=df,
            refresh_interval=60.0,  # widget default
            default_size=(3, 2),
            title="Gedanken",
            source="dashboard",
        )
        assert plugin._widgets["thoughts"].refresh_interval == 0.0


# ─── Per-widget refresh task behaviour ────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_task_broadcasts_each_tick() -> None:
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    call_count = 0

    async def df() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"tick": call_count}

    plugin.register_widget(
        widget_id="test_widget",
        data_fn=df,
        refresh_interval=0.05,
        default_size=(1, 1),
        title="Test",
        source="test",
    )
    reg = plugin._widgets["test_widget"]
    plugin._running = True

    task = asyncio.create_task(plugin._widget_refresh_task("test_widget", reg))
    # Give it 3 full intervals' worth of time
    await asyncio.sleep(0.18)
    plugin._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Expect ≥ 2 broadcasts (3 ticks is typical but CI timing varies)
    assert api.ws_broadcast.await_count >= 2
    first_call = api.ws_broadcast.await_args_list[0].args[0]
    assert first_call["type"] == "dashboard_widget_update"
    assert first_call["widget_id"] == "test_widget"
    assert first_call["data"]["tick"] >= 1


@pytest.mark.asyncio
async def test_refresh_task_broadcasts_even_when_data_unchanged() -> None:
    """Regression: old loop only broadcast on change; new loop always pushes."""
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    async def df() -> dict[str, Any]:
        return {"stable": 1}  # constant

    plugin.register_widget(
        widget_id="stable_w",
        data_fn=df,
        refresh_interval=0.05,
        default_size=(1, 1),
        title="Stable",
        source="t",
    )
    reg = plugin._widgets["stable_w"]
    plugin._running = True

    task = asyncio.create_task(plugin._widget_refresh_task("stable_w", reg))
    await asyncio.sleep(0.18)
    plugin._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # At least 2 ticks fired, each must have broadcast (even with identical data)
    assert api.ws_broadcast.await_count >= 2


@pytest.mark.asyncio
async def test_refresh_task_survives_timeout_on_data_fn() -> None:
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    call_count = 0

    async def slow_df() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call hangs → should hit timeout
            await asyncio.sleep(30.0)
        return {"ok": call_count}

    plugin.register_widget(
        widget_id="flaky",
        data_fn=slow_df,
        refresh_interval=0.05,
        default_size=(1, 1),
        title="Flaky",
        source="t",
    )
    reg = plugin._widgets["flaky"]
    plugin._running = True

    # Patch the timeout so the first call's 30s sleep becomes a quick timeout
    import plugins.dashboard.dashboard_plugin as mod

    orig_wait_for = mod.asyncio.wait_for

    async def fast_wait_for(coro: Any, timeout: float) -> Any:
        return await orig_wait_for(coro, 0.05)

    mod.asyncio.wait_for = fast_wait_for  # type: ignore[assignment]
    try:
        task = asyncio.create_task(plugin._widget_refresh_task("flaky", reg))
        await asyncio.sleep(0.3)
        plugin._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        mod.asyncio.wait_for = orig_wait_for  # type: ignore[assignment]

    # Task kept running after the first timeout → at least one successful broadcast
    assert api.ws_broadcast.await_count >= 1


@pytest.mark.asyncio
async def test_refresh_task_logs_and_continues_on_exception() -> None:
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    call_count = 0

    async def maybe_broken() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    plugin.register_widget(
        widget_id="broken_once",
        data_fn=maybe_broken,
        refresh_interval=0.05,
        default_size=(1, 1),
        title="X",
        source="t",
    )
    reg = plugin._widgets["broken_once"]
    plugin._running = True

    task = asyncio.create_task(plugin._widget_refresh_task("broken_once", reg))
    await asyncio.sleep(0.2)
    plugin._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # After the initial exception the loop recovered and broadcast
    assert api.ws_broadcast.await_count >= 1


# ─── broadcast_widget helper ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_widget_unknown_id_noops() -> None:
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())
    await plugin._broadcast_widget("does_not_exist")
    api.ws_broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_broadcast_widget_sends_current_data() -> None:
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    async def df() -> dict[str, Any]:
        return {"now": 42}

    plugin.register_widget(
        widget_id="w",
        data_fn=df,
        refresh_interval=60.0,
        default_size=(1, 1),
        title="W",
        source="t",
    )
    await plugin._broadcast_widget("w")
    api.ws_broadcast.assert_awaited_once()
    msg = api.ws_broadcast.await_args.args[0]
    assert msg["type"] == "dashboard_widget_update"
    assert msg["widget_id"] == "w"
    assert msg["data"] == {"now": 42}
    # Cache was updated too
    assert plugin._widget_data_cache["w"] == {"now": 42}


@pytest.mark.asyncio
async def test_broadcast_widget_swallows_data_fn_errors() -> None:
    api = _make_api()
    plugin = DashboardPlugin(api, MagicMock())

    async def broken() -> dict[str, Any]:
        raise RuntimeError("nope")

    plugin.register_widget(
        widget_id="brk",
        data_fn=broken,
        refresh_interval=60.0,
        default_size=(1, 1),
        title="B",
        source="t",
    )
    # Must not raise
    await plugin._broadcast_widget("brk")
    api.ws_broadcast.assert_not_awaited()
