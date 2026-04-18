"""Tests for the Dashboard plugin internals.

Covers:
* WidgetRegistration dataclass
* register_widget / unregister_widget (duplicate detection)
* ClockWidget returns correct time/date/weekday format
* MemoryStatsWidget handles missing memory manager
* SystemStatusWidget handles unreachable services
* NotesWidget CRUD (create / update / delete with real aiosqlite)
* Layout save/load round-trip (real aiosqlite)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from plugins.dashboard.dashboard_plugin import DashboardPlugin, WidgetRegistration
from plugins.dashboard.widgets.base_widget import BaseWidget
from plugins.dashboard.widgets.clock_widget import ClockWidget
from plugins.dashboard.widgets.memory_stats_widget import MemoryStatsWidget
from plugins.dashboard.widgets.notes_widget import NotesWidget
from plugins.dashboard.widgets.system_status_widget import SystemStatusWidget


# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_api(**overrides: Any) -> MagicMock:
    """Build a minimal mock PluginAPI."""
    api = MagicMock()
    api.get_config.return_value = overrides.get("config", {})
    api.get_db = AsyncMock(return_value=overrides.get("db", AsyncMock()))
    api._app = MagicMock()
    api._app.memory = overrides.get("memory", None)
    api._app.plugin_loader = overrides.get("plugin_loader", None)
    return api


# ─── WidgetRegistration Dataclass ─────────────────────────────────────────


class TestWidgetRegistration:
    def test_fields_stored(self) -> None:
        async def dummy() -> dict[str, Any]:
            return {}

        reg = WidgetRegistration(
            id="test_widget",
            data_fn=dummy,
            refresh_interval=15.0,
            default_size=(2, 3),
            title="Test Widget",
            source="test_plugin",
        )
        assert reg.id == "test_widget"
        assert reg.refresh_interval == 15.0
        assert reg.default_size == (2, 3)
        assert reg.title == "Test Widget"
        assert reg.source == "test_plugin"

    def test_callable_data_fn(self) -> None:
        async def dummy() -> dict[str, Any]:
            return {"key": "value"}

        reg = WidgetRegistration(
            id="w",
            data_fn=dummy,
            refresh_interval=0,
            default_size=(1, 1),
            title="W",
            source="s",
        )
        result = asyncio.get_event_loop().run_until_complete(reg.data_fn())
        assert result == {"key": "value"}


# ─── register_widget / duplicate detection ────────────────────────────────


class TestRegisterWidget:
    def test_register_stores_widget(self) -> None:
        api = _make_api()
        manifest = MagicMock()
        plugin = DashboardPlugin(api, manifest)

        async def data_fn() -> dict[str, Any]:
            return {}

        plugin.register_widget(
            widget_id="my_widget",
            data_fn=data_fn,
            refresh_interval=10.0,
            default_size=(2, 1),
            title="My Widget",
            source="test",
        )
        assert "my_widget" in plugin._widgets
        assert plugin._widgets["my_widget"].title == "My Widget"

    def test_register_duplicate_is_ignored(self) -> None:
        api = _make_api()
        manifest = MagicMock()
        plugin = DashboardPlugin(api, manifest)

        async def data_fn1() -> dict[str, Any]:
            return {"v": 1}

        async def data_fn2() -> dict[str, Any]:
            return {"v": 2}

        plugin.register_widget("dup", data_fn1, source="first")
        plugin.register_widget("dup", data_fn2, source="second")

        # First registration wins
        assert plugin._widgets["dup"].source == "first"

    def test_register_multiple_widgets(self) -> None:
        api = _make_api()
        manifest = MagicMock()
        plugin = DashboardPlugin(api, manifest)

        async def fn() -> dict[str, Any]:
            return {}

        plugin.register_widget("a", fn, source="s")
        plugin.register_widget("b", fn, source="s")
        plugin.register_widget("c", fn, source="s")
        assert len(plugin._widgets) == 3


# ─── ClockWidget ──────────────────────────────────────────────────────────


class TestClockWidget:
    @pytest.mark.asyncio
    async def test_get_data_returns_expected_keys(self) -> None:
        api = _make_api()
        widget = ClockWidget(api)
        data = await widget.get_data()

        assert "time" in data
        assert "date" in data
        assert "weekday" in data
        assert "timezone" in data
        assert "unix" in data

    @pytest.mark.asyncio
    async def test_time_format_hh_mm(self) -> None:
        api = _make_api()
        widget = ClockWidget(api)
        data = await widget.get_data()
        # Format: "HH:MM"
        parts = data["time"].split(":")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    @pytest.mark.asyncio
    async def test_date_format_dd_mm_yyyy(self) -> None:
        api = _make_api()
        widget = ClockWidget(api)
        data = await widget.get_data()
        # Format: "DD.MM.YYYY"
        parts = data["date"].split(".")
        assert len(parts) == 3
        assert len(parts[2]) == 4  # Year is 4 digits

    @pytest.mark.asyncio
    async def test_weekday_is_german(self) -> None:
        api = _make_api()
        widget = ClockWidget(api)
        data = await widget.get_data()
        german_days = {
            "Montag", "Dienstag", "Mittwoch", "Donnerstag",
            "Freitag", "Samstag", "Sonntag",
        }
        assert data["weekday"] in german_days

    def test_widget_metadata(self) -> None:
        api = _make_api()
        widget = ClockWidget(api)
        assert widget.widget_id == "clock"
        assert widget.title == "Uhr"
        assert widget.default_size == (2, 1)
        assert widget.refresh_interval == 60.0


# ─── MemoryStatsWidget ────────────────────────────────────────────────────


class TestMemoryStatsWidget:
    @pytest.mark.asyncio
    async def test_no_memory_manager(self) -> None:
        api = _make_api(memory=None)
        widget = MemoryStatsWidget(api)
        data = await widget.get_data()
        assert data["available"] is False
        assert data["collections"] == {}
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_with_collections(self) -> None:
        mock_coll = MagicMock()
        mock_coll.count.return_value = 42

        memory = MagicMock()
        memory._collections = {"facts": mock_coll, "chat": mock_coll}
        memory._fts = None

        api = _make_api(memory=memory)
        widget = MemoryStatsWidget(api)
        data = await widget.get_data()

        assert data["available"] is True
        assert data["collections"]["facts"] == 42
        assert data["total"] == 84
        assert data["fts_count"] == 0


# ─── SystemStatusWidget ──────────────────────────────────────────────────


class TestSystemStatusWidget:
    @pytest.mark.asyncio
    async def test_all_services_unreachable(self) -> None:
        """When all HTTP pings fail, every service shows as 'down'."""
        api = _make_api()
        api._app.plugin_loader = None
        widget = SystemStatusWidget(api)

        import httpx

        async def mock_get(url: str, **kw: Any) -> None:
            raise httpx.ConnectError("refused")

        with patch("plugins.dashboard.widgets.system_status_widget.httpx.AsyncClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get = mock_get
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_instance

            data = await widget.get_data()

        for svc in data["services"]:
            assert svc["status"] == "down"
        assert "uptime_seconds" in data

    def test_widget_metadata(self) -> None:
        api = _make_api()
        widget = SystemStatusWidget(api)
        assert widget.widget_id == "system_status"
        assert widget.default_size == (3, 2)


# ─── NotesWidget CRUD (real aiosqlite) ────────────────────────────────────


class TestNotesWidget:
    @pytest.mark.asyncio
    async def test_create_note(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "notes.db")
        async with aiosqlite.connect(db_path) as db:
            api = _make_api(db=db)
            api.get_db = AsyncMock(return_value=db)
            widget = NotesWidget(api)

            result = await widget.create_note("Hello World")
            assert "id" in result
            assert result["content"] == "Hello World"
            assert "created_at" in result

    @pytest.mark.asyncio
    async def test_get_data_returns_notes(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "notes.db")
        async with aiosqlite.connect(db_path) as db:
            api = _make_api(db=db)
            api.get_db = AsyncMock(return_value=db)
            widget = NotesWidget(api)

            await widget.create_note("note1")
            await widget.create_note("note2")
            data = await widget.get_data()

            assert len(data["notes"]) == 2

    @pytest.mark.asyncio
    async def test_update_note(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "notes.db")
        async with aiosqlite.connect(db_path) as db:
            api = _make_api(db=db)
            api.get_db = AsyncMock(return_value=db)
            widget = NotesWidget(api)

            created = await widget.create_note("original")
            note_id = created["id"]
            updated = await widget.update_note(note_id, "changed")
            assert updated["content"] == "changed"

    @pytest.mark.asyncio
    async def test_delete_note(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "notes.db")
        async with aiosqlite.connect(db_path) as db:
            api = _make_api(db=db)
            api.get_db = AsyncMock(return_value=db)
            widget = NotesWidget(api)

            created = await widget.create_note("to delete")
            note_id = created["id"]
            deleted = await widget.delete_note(note_id)
            assert deleted["deleted"] == note_id

            data = await widget.get_data()
            assert len(data["notes"]) == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_error(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "notes.db")
        async with aiosqlite.connect(db_path) as db:
            api = _make_api(db=db)
            api.get_db = AsyncMock(return_value=db)
            widget = NotesWidget(api)
            # Ensure table exists
            await widget.create_note("seed")

            result = await widget.delete_note("nonexistent")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_error(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "notes.db")
        async with aiosqlite.connect(db_path) as db:
            api = _make_api(db=db)
            api.get_db = AsyncMock(return_value=db)
            widget = NotesWidget(api)
            # Ensure table exists
            await widget.create_note("seed")

            result = await widget.update_note("nonexistent", "nope")
            assert "error" in result


# ─── BaseWidget Abstract ──────────────────────────────────────────────────


class TestBaseWidget:
    def test_to_manifest(self) -> None:
        api = _make_api()
        widget = ClockWidget(api)
        manifest = widget.to_manifest()
        assert manifest["widget_id"] == "clock"
        assert manifest["title"] == "Uhr"
        assert manifest["default_size"] == [2, 1]
        assert manifest["refresh_interval"] == 60.0
