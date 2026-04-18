"""
Integration-style tests for the Phase-6 SchedulerPlugin extension.

These tests stand up a real :class:`SchedulerPlugin` against a stub
PluginAPI backed by an in-memory ``aiosqlite`` connection. They cover:

* Schema migration (legacy rows without the new columns survive)
* The five schedule modes (timer / reminder / recurring / proactive_chat /
  agent_task) go through the shared ``_insert_timer`` path and land in
  the DB with the correct action blob
* ``_tick`` dispatches the right side-effect per action type:
  notify → ws_broadcast, proactive_chat → api.agent_proactive,
  agent_task → skill_writer.AgentManager.spawn, tool → registry.execute
* A recurring timer reschedules itself after firing (``fire_at`` moves
  forward, ``fired`` stays 0)
* A one-shot proactive_chat timer sets ``fired=1`` after firing

We bypass the real PluginAPI by constructing a lightweight fake that
exposes only the surface :class:`SchedulerPlugin` touches — keeps the
tests fast and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time as _time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
import pytest_asyncio

from plugins.scheduler.scheduler_plugin import (
    SchedulerPlugin,
    _decode_action,
    _encode_action,
    _parse_hhmm_to_timestamp,
)


# ─── Fakes ──────────────────────────────────────────────────────────────────


class _FakePluginAPI:
    """Minimal fake of :class:`PluginAPI` for scheduler tests."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._config: dict[str, Any] = {
            "check_interval": 5.0,
            "max_active_timers": 50,
            "enable_impulses": False,
        }
        self.tools: dict[str, Any] = {}
        self.ws_handlers: dict[str, Any] = {}
        self.broadcasts: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.agent_proactive_calls: list[dict[str, Any]] = []
        self.agent_proactive_result: bool = True
        self.skill_writer_plugin: Any = None
        self.tool_registry: Any = None

    async def get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(str(self._db_path))
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.commit()
        return self._db

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def register_tool(
        self, name: str, handler: Any, description: str, schema: dict[str, Any]
    ) -> None:
        self.tools[name] = {
            "handler": handler,
            "description": description,
            "schema": schema,
        }

    def register_ws_handler(self, msg_type: str, handler: Any) -> None:
        self.ws_handlers[msg_type] = handler

    async def ws_broadcast(self, data: dict[str, Any]) -> None:
        self.broadcasts.append(dict(data))

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        self.events.append((event, dict(data or {})))

    async def agent_proactive(
        self, session_id: str, prompt: str, label: str = ""
    ) -> bool:
        self.agent_proactive_calls.append({
            "session_id": session_id,
            "prompt": prompt,
            "label": label,
        })
        return self.agent_proactive_result

    def get_plugin(self, name: str) -> Any:
        if name == "skill_writer":
            return self.skill_writer_plugin
        return None

    def get_tool_registry(self) -> Any:
        return self.tool_registry

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def plugin_with_api():
    tmp = tempfile.mkdtemp(prefix="lexy_sched_test_")
    db_path = Path(tmp) / "scheduler.db"
    api = _FakePluginAPI(db_path)

    manifest = MagicMock()
    manifest.config_defaults = {}
    plugin = SchedulerPlugin(api=api, manifest=manifest)

    # Mimic on_load + on_enable manually so we bypass background loop.
    await plugin.on_load()

    yield plugin, api

    # Cleanup
    await api.close()
    try:
        os.remove(db_path)
    except OSError:
        pass
    try:
        os.remove(str(db_path) + "-wal")
    except OSError:
        pass
    try:
        os.remove(str(db_path) + "-shm")
    except OSError:
        pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass


# ─── Helpers ────────────────────────────────────────────────────────────────


async def _count_rows(api: _FakePluginAPI) -> int:
    db = await api.get_db()
    cursor = await db.execute("SELECT COUNT(*) FROM timers")
    row = await cursor.fetchone()
    await cursor.close()
    return int(row[0]) if row else 0


async def _get_row(api: _FakePluginAPI, timer_id: str) -> dict[str, Any] | None:
    db = await api.get_db()
    cursor = await db.execute(
        "SELECT id, kind, label, fire_at, fired, cancelled, "
        "repeat_pattern, repeat_interval, action, project_id, "
        "last_fired_at, active FROM timers WHERE id = ?",
        (timer_id,),
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "kind": row[1],
        "label": row[2],
        "fire_at": row[3],
        "fired": bool(row[4]),
        "cancelled": bool(row[5]),
        "repeat_pattern": row[6] or "",
        "repeat_interval": int(row[7] or 0),
        "action": row[8] or "",
        "project_id": row[9] or "default",
        "last_fired_at": row[10] or 0.0,
        "active": bool(row[11]),
    }


# ─── Encode / decode helpers ────────────────────────────────────────────────


class TestActionHelpers:
    def test_notify_encodes_empty(self) -> None:
        assert _encode_action("notify", {"ignored": True}) == ""

    def test_proactive_encode_roundtrip(self) -> None:
        blob = _encode_action("proactive_chat", {"session_id": "s1", "prompt": "hi"})
        data = _decode_action(blob)
        assert data["type"] == "proactive_chat"
        assert data["session_id"] == "s1"

    def test_decode_empty_string(self) -> None:
        assert _decode_action("") == {}

    def test_decode_invalid_json(self) -> None:
        assert _decode_action("{not json") == {}

    def test_decode_non_object_json(self) -> None:
        assert _decode_action("42") == {}


# ─── HH:MM parser ───────────────────────────────────────────────────────────


class TestHHMMParser:
    def test_past_hhmm_shifts_to_tomorrow(self) -> None:
        from datetime import datetime, timedelta
        past = (datetime.now() - timedelta(hours=2)).strftime("%H:%M")
        ts = _parse_hhmm_to_timestamp(past, tomorrow=False)
        assert ts > _time.time()

    def test_tomorrow_flag_forces_future(self) -> None:
        ts = _parse_hhmm_to_timestamp("06:00", tomorrow=True)
        assert ts > _time.time()

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_hhmm_to_timestamp("nope", tomorrow=False)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_hhmm_to_timestamp("25:99", tomorrow=False)


# ─── Schema migration ───────────────────────────────────────────────────────


class TestSchemaMigration:
    @pytest.mark.asyncio
    async def test_fresh_db_has_all_phase6_columns(
        self, plugin_with_api
    ) -> None:
        _, api = plugin_with_api
        db = await api.get_db()
        cursor = await db.execute("PRAGMA table_info(timers)")
        rows = await cursor.fetchall()
        await cursor.close()
        cols = {r[1] for r in rows}
        assert {
            "repeat_pattern",
            "repeat_interval",
            "action",
            "project_id",
            "last_fired_at",
            "active",
        } <= cols

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        # Run on_load a second time — must not raise.
        await plugin.on_load()
        # Schema is still intact + only one "timers" table.
        db = await api.get_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='timers'"
        )
        row = await cursor.fetchone()
        await cursor.close()
        assert row is not None and row[0] == 1


# ─── Tool handlers (insert + action encoding) ───────────────────────────────


class TestToolInserts:
    @pytest.mark.asyncio
    async def test_set_timer_inserts_row(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_set_timer(label="pasta", minutes=5)
        assert "id" in result
        row = await _get_row(api, result["id"])
        assert row is not None
        assert row["kind"] == "timer"
        assert row["label"] == "pasta"
        assert row["fired"] is False
        assert row["active"] is True
        assert row["repeat_pattern"] == ""
        assert row["action"] == ""

    @pytest.mark.asyncio
    async def test_set_timer_rejects_zero_delay(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        result = await plugin._tool_set_timer(label="x", seconds=0, minutes=0)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_set_reminder_accepts_hhmm(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_set_reminder(
            time="23:59", label="buch", tomorrow=True
        )
        assert "id" in result
        row = await _get_row(api, result["id"])
        assert row is not None
        assert row["kind"] == "reminder"

    @pytest.mark.asyncio
    async def test_set_recurring_stores_pattern(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_set_recurring(
            label="Workout", pattern="mo-fr 18:00"
        )
        assert "id" in result
        row = await _get_row(api, result["id"])
        assert row is not None
        assert row["kind"] == "recurring"
        assert row["repeat_pattern"] == "mo-fr 18:00"

    @pytest.mark.asyncio
    async def test_set_recurring_with_proactive_chat_action(
        self, plugin_with_api
    ) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_set_recurring(
            label="Guten Morgen",
            pattern="daily 09:00",
            action_type="proactive_chat",
            action_payload={"session_id": "s1", "prompt": "Morgenruf"},
        )
        row = await _get_row(api, result["id"])
        assert row is not None
        action = json.loads(row["action"])
        assert action["type"] == "proactive_chat"
        assert action["session_id"] == "s1"
        assert action["prompt"] == "Morgenruf"

    @pytest.mark.asyncio
    async def test_set_recurring_invalid_pattern_returns_error(
        self, plugin_with_api
    ) -> None:
        plugin, _api = plugin_with_api
        result = await plugin._tool_set_recurring(
            label="oops", pattern="kebab 09:00"
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_schedule_proactive_hhmm(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_schedule_proactive(
            label="Check-in",
            time_or_pattern="23:59",
            message="Wie läuft der Tag?",
            session_id="s1",
            tomorrow=True,
        )
        assert "id" in result
        row = await _get_row(api, result["id"])
        assert row is not None
        assert row["kind"] == "proactive_chat"
        action = json.loads(row["action"])
        assert action["type"] == "proactive_chat"
        assert action["prompt"] == "Wie läuft der Tag?"

    @pytest.mark.asyncio
    async def test_schedule_proactive_with_pattern(
        self, plugin_with_api
    ) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_schedule_proactive(
            label="Morgengruß",
            time_or_pattern="daily 09:00",
            message="Guten Morgen Mike",
            session_id="s1",
        )
        row = await _get_row(api, result["id"])
        assert row is not None
        assert row["kind"] == "recurring"
        assert row["repeat_pattern"] == "daily 09:00"

    @pytest.mark.asyncio
    async def test_schedule_agent_task(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        result = await plugin._tool_schedule_agent_task(
            label="News",
            time_or_pattern="daily 08:00",
            persona="researcher",
            task="Fasse die Morgenheadlines zusammen",
            report_to_session="s1",
        )
        row = await _get_row(api, result["id"])
        assert row is not None
        action = json.loads(row["action"])
        assert action["type"] == "agent_task"
        assert action["persona"] == "researcher"
        assert action["report_to_session"] == "s1"


# ─── update_timer ───────────────────────────────────────────────────────────


class TestUpdateTimer:
    @pytest.mark.asyncio
    async def test_update_label_and_active(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        created = await plugin._tool_set_timer(label="orig", seconds=3600)
        tid = created["id"]

        res = await plugin._tool_update_timer(id=tid, label="renamed", active=False)
        assert "error" not in res
        row = await _get_row(api, tid)
        assert row is not None
        assert row["label"] == "renamed"
        assert row["active"] is False

    @pytest.mark.asyncio
    async def test_update_pattern_empty_clears(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        created = await plugin._tool_set_recurring(
            label="L", pattern="every 30m"
        )
        tid = created["id"]
        res = await plugin._tool_update_timer(id=tid, pattern="")
        assert "error" not in res
        row = await _get_row(api, tid)
        assert row is not None
        assert row["repeat_pattern"] == ""
        assert row["repeat_interval"] == 0

    @pytest.mark.asyncio
    async def test_update_invalid_pattern(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        created = await plugin._tool_set_timer(label="x", seconds=60)
        res = await plugin._tool_update_timer(id=created["id"], pattern="kebab")
        assert "error" in res

    @pytest.mark.asyncio
    async def test_update_missing_id(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        res = await plugin._tool_update_timer(id="nonexistent", label="x")
        assert "error" in res

    @pytest.mark.asyncio
    async def test_update_no_fields(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        created = await plugin._tool_set_timer(label="x", seconds=60)
        res = await plugin._tool_update_timer(id=created["id"])
        assert "error" in res


# ─── _tick dispatches correct side-effects ──────────────────────────────────


class TestTickDispatch:
    @pytest.mark.asyncio
    async def test_notify_only_broadcasts_and_fires(
        self, plugin_with_api
    ) -> None:
        plugin, api = plugin_with_api
        created = await plugin._tool_set_timer(label="plain", seconds=60)
        tid = created["id"]
        # Force fire_at into the past.
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 10, tid),
        )
        await db.commit()
        # Clear any startup broadcasts from _insert_timer.
        api.broadcasts.clear()

        await plugin._tick()

        row = await _get_row(api, tid)
        assert row is not None
        assert row["fired"] is True
        types = [b.get("type") for b in api.broadcasts]
        assert "scheduler_triggered" in types

    @pytest.mark.asyncio
    async def test_proactive_chat_fires_agent(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        created = await plugin._tool_schedule_proactive(
            label="Morgenruf",
            time_or_pattern="23:59",
            message="Guten Morgen",
            session_id="s1",
            tomorrow=True,
        )
        tid = created["id"]
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 10, tid),
        )
        await db.commit()
        api.broadcasts.clear()

        await plugin._tick()

        assert len(api.agent_proactive_calls) == 1
        call = api.agent_proactive_calls[0]
        assert call["session_id"] == "s1"
        assert call["prompt"] == "Guten Morgen"
        assert call["label"] == "Morgenruf"

    @pytest.mark.asyncio
    async def test_proactive_chat_fallback_emits_event_on_agent_miss(
        self, plugin_with_api
    ) -> None:
        plugin, api = plugin_with_api
        api.agent_proactive_result = False  # force fallback path

        created = await plugin._tool_schedule_proactive(
            label="Fallback",
            time_or_pattern="23:59",
            message="Testprompt",
            session_id="s1",
            tomorrow=True,
        )
        tid = created["id"]
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 10, tid),
        )
        await db.commit()
        api.events.clear()

        await plugin._tick()

        event_names = [ev[0] for ev in api.events]
        assert "core.scheduler_proactive" in event_names

    @pytest.mark.asyncio
    async def test_agent_task_uses_skill_writer_manager(
        self, plugin_with_api
    ) -> None:
        plugin, api = plugin_with_api

        # Wire a fake skill_writer plugin with an async spawn() method.
        fake_manager = MagicMock()
        fake_manager.spawn = AsyncMock(return_value={
            "agent_id": "abc",
            "name": "sched_researcher",
            "task": "Research",
            "status": "running",
        })
        fake_sw = MagicMock()
        fake_sw._agent_manager = fake_manager
        api.skill_writer_plugin = fake_sw

        created = await plugin._tool_schedule_agent_task(
            label="News",
            time_or_pattern="23:59",
            persona="researcher",
            task="Research",
            report_to_session="s1",
            tomorrow=True,
        )
        tid = created["id"]
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 10, tid),
        )
        await db.commit()
        api.broadcasts.clear()

        await plugin._tick()

        fake_manager.spawn.assert_awaited_once()
        call_kwargs = fake_manager.spawn.await_args.kwargs
        assert call_kwargs["name"] == "sched_researcher"
        assert call_kwargs["task"] == "Research"
        types = [b.get("type") for b in api.broadcasts]
        assert "agent_task_spawned" in types

    @pytest.mark.asyncio
    async def test_agent_task_without_manager_logs_skip(
        self, plugin_with_api
    ) -> None:
        plugin, api = plugin_with_api
        # Do NOT install a skill_writer plugin.
        created = await plugin._tool_schedule_agent_task(
            label="News",
            time_or_pattern="23:59",
            persona="researcher",
            task="Research",
            report_to_session="s1",
            tomorrow=True,
        )
        tid = created["id"]
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 10, tid),
        )
        await db.commit()
        api.broadcasts.clear()

        await plugin._tick()

        types = [b.get("type") for b in api.broadcasts]
        assert "agent_task_skipped" in types


# ─── Recurring reschedules ──────────────────────────────────────────────────


class TestRecurringReschedule:
    @pytest.mark.asyncio
    async def test_every_30m_reschedules_forward(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        created = await plugin._tool_set_recurring(
            label="every30", pattern="every 30m"
        )
        tid = created["id"]
        original_fire = (await _get_row(api, tid))["fire_at"]

        # Move fire_at into the past.
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 5, tid),
        )
        await db.commit()

        await plugin._tick()

        row = await _get_row(api, tid)
        assert row is not None
        assert row["fired"] is False  # stays actionable
        assert row["fire_at"] > _time.time()  # moved into the future
        # last_fired_at updated
        assert row["last_fired_at"] > 0.0

    @pytest.mark.asyncio
    async def test_one_shot_proactive_marks_fired(self, plugin_with_api) -> None:
        plugin, api = plugin_with_api
        created = await plugin._tool_schedule_proactive(
            label="one-shot",
            time_or_pattern="23:59",
            message="hi",
            session_id="s1",
            tomorrow=True,
        )
        tid = created["id"]
        db = await api.get_db()
        await db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?",
            (_time.time() - 5, tid),
        )
        await db.commit()

        await plugin._tick()

        row = await _get_row(api, tid)
        assert row is not None
        assert row["fired"] is True

    @pytest.mark.asyncio
    async def test_bad_pattern_marks_fired_instead_of_looping(
        self, plugin_with_api
    ) -> None:
        """A corrupt pattern in the DB must not cause infinite tick loops."""
        plugin, api = plugin_with_api
        # Insert a row with a bogus pattern directly.
        db = await api.get_db()
        await db.execute(
            """INSERT INTO timers (
                id, kind, label, fire_at, created_at,
                fired, cancelled, session_id,
                repeat_pattern, repeat_interval, action,
                project_id, last_fired_at, active
            ) VALUES (?, 'recurring', 'bad', ?, ?, 0, 0, '',
                      'kebab 09:00', 0, '', 'default', 0, 1)""",
            ("bad1", _time.time() - 5, _time.time()),
        )
        await db.commit()

        await plugin._tick()

        row = await _get_row(api, "bad1")
        assert row is not None
        assert row["fired"] is True  # Stopped retrying.


# ─── list_timers / cancel_timer ─────────────────────────────────────────────


class TestListAndCancel:
    @pytest.mark.asyncio
    async def test_list_active_only(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        await plugin._tool_set_timer(label="a", seconds=60)
        b = await plugin._tool_set_timer(label="b", seconds=60)
        await plugin._tool_cancel_timer(id=b["id"])

        listing = await plugin._tool_list_timers()
        ids = [t["id"] for t in listing["timers"]]
        assert b["id"] not in ids  # cancelled excluded by default

    @pytest.mark.asyncio
    async def test_list_include_inactive(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        a = await plugin._tool_set_timer(label="a", seconds=60)
        await plugin._tool_update_timer(id=a["id"], active=False)
        listing = await plugin._tool_list_timers(include_inactive=True)
        ids = [t["id"] for t in listing["timers"]]
        assert a["id"] in ids


# ─── WS create handler ─────────────────────────────────────────────────────


class TestWSCreate:
    @pytest.mark.asyncio
    async def test_ws_create_recurring(self, plugin_with_api) -> None:
        plugin, _api = plugin_with_api
        client = MagicMock()
        client.send_json = AsyncMock()
        await plugin._handle_ws_create(
            client,
            {
                "mode": "recurring",
                "label": "Morgenruf",
                "pattern": "daily 09:00",
                "action_type": "notify",
            },
        )
        client.send_json.assert_awaited_once()
        payload = client.send_json.await_args.args[0]
        assert payload["type"] == "scheduler_created"
        assert payload["mode"] == "recurring"
        assert "id" in payload

    @pytest.mark.asyncio
    async def test_ws_create_bad_mode_returns_error(
        self, plugin_with_api
    ) -> None:
        plugin, _api = plugin_with_api
        client = MagicMock()
        client.send_json = AsyncMock()
        await plugin._handle_ws_create(
            client, {"mode": "flurble", "label": "Oops"}
        )
        payload = client.send_json.await_args.args[0]
        assert "error" in payload
