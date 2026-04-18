"""
Lexy AI - Scheduler Plugin.

Schlanke v2-Neuauflage des v1-Schedulers. Bietet:

* **Timer** – "in N Minuten"/"in N Sekunden"
* **Reminders** – zu einer absoluten Uhrzeit heute/morgen
* **Recurring** – wiederkehrende Schedules ("daily 09:00", "every 30m",
  "mo-fr 18:00", "weekly mo 14:00", "monthly 1 09:00")
* **Proactive-Chat** – Lexy meldet sich aktiv in einer Session ("sprich
  Mike an") ausgelöst durch einen Timer oder ein Recurring-Pattern
* **Agent-Task** – Scheduler startet einen :class:`AutoAgent` mit einer
  Aufgabe und meldet dessen Ergebnis in eine Session zurück
* **Impulses** – optionale zufällige Daydream-Impulse (ruft den Agent mit
  einer Situationsbeschreibung auf, damit Lexy von sich aus spricht)

Features:

* Persistenz via aiosqlite (Plugin-eigene DB, automatische Schema-Migration).
* Background-Loop checkt alle ``check_interval`` Sekunden.
* LLM-Tools registriert, damit Gemma 4 Timer selbst setzen kann:
  ``set_timer``, ``set_reminder``, ``set_recurring``,
  ``schedule_proactive_reminder``, ``schedule_agent_task``, ``update_timer``,
  ``list_timers``, ``cancel_timer``.
* WebSocket-Broadcast ``scheduler.triggered`` bei Auslösung.
* EventBus ``core.scheduler_triggered`` für andere Plugins.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .recurring_parser import (
    RecurringSpec,
    next_fire_at,
    parse_recurring,
)

log = get_logger(module="scheduler_plugin")


# ─── Tool schemas ───────────────────────────────────────────────────────────

SET_TIMER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seconds": {
            "type": "integer",
            "description": "Delay in seconds (use this OR minutes, not both)",
        },
        "minutes": {
            "type": "integer",
            "description": "Delay in minutes",
        },
        "label": {
            "type": "string",
            "description": "Short label for the timer (e.g. 'Pasta fertig')",
        },
    },
    "required": ["label"],
}

SET_REMINDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "time": {
            "type": "string",
            "description": "HH:MM 24h format for today or tomorrow",
        },
        "label": {
            "type": "string",
            "description": "What the reminder is about",
        },
        "tomorrow": {
            "type": "boolean",
            "description": "If true, schedule for tomorrow instead of today",
        },
    },
    "required": ["time", "label"],
}

SET_RECURRING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "Human-readable label for this recurring timer.",
        },
        "pattern": {
            "type": "string",
            "description": (
                "Recurring pattern. Examples: 'daily 09:00', 'every 30m', "
                "'mo-fr 18:00', 'weekly mo 14:00', 'monthly 1 09:00'."
            ),
        },
        "action_type": {
            "type": "string",
            "description": (
                "Optional action kind when the timer fires. One of 'notify' "
                "(default, just broadcasts a toast), 'proactive_chat' "
                "(Lexy schreibt aktiv in einer Session), 'agent_task' "
                "(startet einen AutoAgent)."
            ),
        },
        "action_payload": {
            "type": "object",
            "description": (
                "Action parameters. For 'proactive_chat' expects "
                "{session_id, prompt}. For 'agent_task' expects "
                "{persona, task, report_to_session}."
            ),
        },
    },
    "required": ["label", "pattern"],
}

SCHEDULE_PROACTIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Short label"},
        "time_or_pattern": {
            "type": "string",
            "description": (
                "Either HH:MM (one-shot today/tomorrow), or a recurring "
                "pattern string such as 'daily 09:00' or 'mo-fr 18:00'."
            ),
        },
        "message": {
            "type": "string",
            "description": (
                "What Lexy should say / be reminded of. This is the internal "
                "prompt handed to process_proactive()."
            ),
        },
        "session_id": {
            "type": "string",
            "description": "Session to write into. Defaults to the current session.",
        },
        "tomorrow": {
            "type": "boolean",
            "description": "For HH:MM input only: force schedule to tomorrow.",
        },
    },
    "required": ["label", "time_or_pattern", "message"],
}

SCHEDULE_AGENT_TASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "description": "Short label"},
        "time_or_pattern": {
            "type": "string",
            "description": "HH:MM or a recurring pattern string.",
        },
        "persona": {
            "type": "string",
            "description": "Persona / name of the sub-agent (e.g. 'researcher').",
        },
        "task": {
            "type": "string",
            "description": "The task prompt for the sub-agent.",
        },
        "report_to_session": {
            "type": "string",
            "description": "Session to broadcast the agent result into.",
        },
        "tomorrow": {
            "type": "boolean",
            "description": "For HH:MM input only: force schedule to tomorrow.",
        },
    },
    "required": ["label", "time_or_pattern", "task"],
}

UPDATE_TIMER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Timer id to update."},
        "label": {"type": "string"},
        "fire_at_iso": {
            "type": "string",
            "description": "ISO-8601 timestamp to move the next fire to.",
        },
        "pattern": {
            "type": "string",
            "description": "Replace the recurring pattern. Empty string disables recurrence.",
        },
        "active": {
            "type": "boolean",
            "description": "Set to false to pause, true to resume.",
        },
    },
    "required": ["id"],
}

LIST_TIMERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_inactive": {"type": "boolean"},
    },
}

CANCEL_TIMER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "The timer id to cancel"},
    },
    "required": ["id"],
}


# ─── Plugin ─────────────────────────────────────────────────────────────────


class SchedulerPlugin(BasePlugin):
    """Timer / reminder / alarm / recurring / agent-task scheduler."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._check_interval: float = 5.0
        self._max_active: int = 50
        self._loop_task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._enable_impulses: bool = False
        self._impulse_min_hour: int = 8
        self._impulse_max_hour: int = 22
        self._next_impulse_at: float | None = None

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        self._apply_config(self.api.get_config())
        await self._init_db()

    async def on_config_changed(self, cfg: dict[str, Any]) -> None:
        """Settings editor re-applies the config at runtime (Phase 8)."""
        self._apply_config(cfg)
        log.info(
            "scheduler.config_changed",
            interval=self._check_interval,
            max_active=self._max_active,
            impulses=self._enable_impulses,
        )

    def _apply_config(self, config: dict[str, Any]) -> None:
        try:
            self._check_interval = float(config.get("check_interval", 5.0))
        except (TypeError, ValueError):
            self._check_interval = 5.0
        try:
            self._max_active = int(config.get("max_active_timers", 50))
        except (TypeError, ValueError):
            self._max_active = 50
        self._enable_impulses = bool(config.get("enable_impulses", False))
        try:
            self._impulse_min_hour = int(config.get("impulse_min_hour", 8))
        except (TypeError, ValueError):
            self._impulse_min_hour = 8
        try:
            self._impulse_max_hour = int(config.get("impulse_max_hour", 22))
        except (TypeError, ValueError):
            self._impulse_max_hour = 22

    async def _init_db(self) -> None:
        db = await self.api.get_db()
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS timers (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                label TEXT NOT NULL,
                fire_at REAL NOT NULL,
                created_at REAL NOT NULL,
                fired INTEGER NOT NULL DEFAULT 0,
                cancelled INTEGER NOT NULL DEFAULT 0,
                session_id TEXT DEFAULT ''
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_timers_fire_at "
            "ON timers(fire_at, fired, cancelled)"
        )

        # Additive schema migration for Phase 6 (active scheduler). Each
        # ALTER is guarded so re-running on an already-migrated DB is a
        # no-op. SQLite raises OperationalError when a column already
        # exists — we catch that and move on.
        for sql in (
            "ALTER TABLE timers ADD COLUMN repeat_pattern TEXT DEFAULT ''",
            "ALTER TABLE timers ADD COLUMN repeat_interval INTEGER DEFAULT 0",
            "ALTER TABLE timers ADD COLUMN action TEXT DEFAULT ''",
            "ALTER TABLE timers ADD COLUMN project_id TEXT DEFAULT 'default'",
            "ALTER TABLE timers ADD COLUMN last_fired_at REAL DEFAULT 0",
            "ALTER TABLE timers ADD COLUMN active INTEGER DEFAULT 1",
        ):
            try:
                await db.execute(sql)
            except Exception:  # noqa: BLE001 — duplicate column is expected
                pass

        await db.commit()

    async def on_enable(self) -> None:
        # Register LLM tools so Gemma can set timers on its own.
        self.api.register_tool(
            name="set_timer",
            handler=self._tool_set_timer,
            description=(
                "Setze einen einfachen Timer. Nutze 'seconds' ODER 'minutes', "
                "und gib einen kurzen Label mit was der Timer bedeutet. "
                "Lexy meldet sich akustisch und im Chat wenn der Timer abläuft."
            ),
            schema=SET_TIMER_SCHEMA,
        )
        self.api.register_tool(
            name="set_reminder",
            handler=self._tool_set_reminder,
            description=(
                "Setze eine Erinnerung zu einer bestimmten Uhrzeit im Format "
                "HH:MM. Optional kann 'tomorrow: true' gesetzt werden."
            ),
            schema=SET_REMINDER_SCHEMA,
        )
        self.api.register_tool(
            name="set_recurring",
            handler=self._tool_set_recurring,
            description=(
                "Registriert einen wiederkehrenden Scheduler-Eintrag. "
                "Unterstützte Patterns: 'daily 09:00', 'every 30m', "
                "'mo-fr 18:00', 'weekly mo 14:00', 'monthly 1 09:00'. "
                "Optional: action_type ('notify'|'proactive_chat'|'agent_task') "
                "mit action_payload."
            ),
            schema=SET_RECURRING_SCHEMA,
        )
        self.api.register_tool(
            name="schedule_proactive_reminder",
            handler=self._tool_schedule_proactive,
            description=(
                "Plant eine proaktive Nachricht von Lexy in einer Session. "
                "time_or_pattern ist entweder eine Uhrzeit (HH:MM) oder ein "
                "Recurring-Pattern. Lexy meldet sich selbst — keine User-Message."
            ),
            schema=SCHEDULE_PROACTIVE_SCHEMA,
        )
        self.api.register_tool(
            name="schedule_agent_task",
            handler=self._tool_schedule_agent_task,
            description=(
                "Plant eine Sub-Agent-Aufgabe. time_or_pattern ist entweder "
                "HH:MM oder ein Recurring-Pattern. Zum geplanten Zeitpunkt "
                "startet der Scheduler einen AutoAgent mit der Task und "
                "meldet das Ergebnis per WebSocket."
            ),
            schema=SCHEDULE_AGENT_TASK_SCHEMA,
        )
        self.api.register_tool(
            name="update_timer",
            handler=self._tool_update_timer,
            description=(
                "Ändert einen existierenden Timer. Erlaubt Label, neuen "
                "fire_at (ISO-8601), neuen Pattern, active pausieren/fortsetzen."
            ),
            schema=UPDATE_TIMER_SCHEMA,
        )
        self.api.register_tool(
            name="list_timers",
            handler=self._tool_list_timers,
            description="Liste alle aktiven Timer und Erinnerungen.",
            schema=LIST_TIMERS_SCHEMA,
        )
        self.api.register_tool(
            name="cancel_timer",
            handler=self._tool_cancel_timer,
            description="Bricht einen aktiven Timer anhand seiner id ab.",
            schema=CANCEL_TIMER_SCHEMA,
        )

        # WebSocket handlers for the GUI.
        self.api.register_ws_handler("scheduler_list", self._handle_ws_list)
        self.api.register_ws_handler("scheduler_cancel", self._handle_ws_cancel)
        self.api.register_ws_handler("scheduler_create", self._handle_ws_create)
        self.api.register_ws_handler("scheduler_update", self._handle_ws_update)

        # Background loop
        self._running = True
        self._loop_task = asyncio.create_task(self._loop(), name="scheduler.loop")

        if self._enable_impulses:
            self._next_impulse_at = self._pick_next_impulse()
            log.info("scheduler.impulses_enabled", next=self._next_impulse_at)

        log.info("scheduler.enabled", check_interval=self._check_interval)

    async def on_disable(self) -> None:
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._loop_task = None
        log.info("scheduler.disabled")

    # ─── Tool handlers ──────────────────────────────────────────────

    async def _tool_set_timer(
        self,
        label: str,
        seconds: int | None = None,
        minutes: int | None = None,
    ) -> dict[str, Any]:
        delay_seconds = 0
        if seconds is not None:
            delay_seconds += int(seconds)
        if minutes is not None:
            delay_seconds += int(minutes) * 60
        if delay_seconds <= 0:
            return {"error": "timer delay must be positive"}
        if delay_seconds > 7 * 24 * 3600:
            return {"error": "timer delay must not exceed 7 days"}

        fire_at = time.time() + delay_seconds
        timer_id = uuid.uuid4().hex[:8]
        await self._insert_timer(
            timer_id=timer_id,
            kind="timer",
            label=label,
            fire_at=fire_at,
        )
        return {
            "id": timer_id,
            "label": label,
            "fires_in_seconds": delay_seconds,
            "fires_at": datetime.fromtimestamp(fire_at).strftime("%H:%M:%S"),
        }

    async def _tool_set_reminder(
        self,
        time: str,  # noqa: A002 — "time" is the parameter name in the LLM schema
        label: str,
        tomorrow: bool = False,
    ) -> dict[str, Any]:
        try:
            fire_at = _parse_hhmm_to_timestamp(time, tomorrow)
        except ValueError as exc:
            return {"error": str(exc)}

        timer_id = uuid.uuid4().hex[:8]
        await self._insert_timer(
            timer_id=timer_id,
            kind="reminder",
            label=label,
            fire_at=fire_at,
        )
        return {
            "id": timer_id,
            "label": label,
            "fires_at": datetime.fromtimestamp(fire_at).strftime("%Y-%m-%d %H:%M"),
        }

    async def _tool_set_recurring(
        self,
        label: str,
        pattern: str,
        action_type: str = "notify",
        action_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            spec = parse_recurring(pattern)
        except ValueError as exc:
            return {"error": f"invalid pattern: {exc}"}

        if action_type not in (
            "notify",
            "proactive_chat",
            "agent_task",
            "tool",
            # character_chat plugin: when the timer fires, character_chat
            # picks it up via ``core.scheduler_triggered`` and runs a
            # round with pulse_from_id/pulse_text from the action payload.
            "character_pulse",
            # character_chat plugin (autonomous simulation): each tick
            # one speaker (Lexy or a character) reacts to the last turn.
            # Payload: {session_id, interval_seconds}.
            "autonomous_sim",
        ):
            return {"error": f"unknown action_type: {action_type!r}"}

        fire_at = next_fire_at(spec, datetime.now()).timestamp()
        timer_id = uuid.uuid4().hex[:8]
        action_blob = _encode_action(action_type, action_payload or {})

        await self._insert_timer(
            timer_id=timer_id,
            kind="recurring",
            label=label,
            fire_at=fire_at,
            repeat_pattern=pattern,
            repeat_interval=spec.interval_seconds if spec.is_interval else 0,
            action=action_blob,
        )
        return {
            "id": timer_id,
            "label": label,
            "pattern": pattern,
            "action_type": action_type,
            "next_fire_at": datetime.fromtimestamp(fire_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

    async def _tool_schedule_proactive(
        self,
        label: str,
        time_or_pattern: str,
        message: str,
        session_id: str = "",
        tomorrow: bool = False,
    ) -> dict[str, Any]:
        return await self._insert_time_or_pattern(
            label=label,
            time_or_pattern=time_or_pattern,
            tomorrow=tomorrow,
            action_type="proactive_chat",
            action_payload={
                "session_id": session_id,
                "prompt": message,
            },
            session_id=session_id,
        )

    async def _tool_schedule_agent_task(
        self,
        label: str,
        time_or_pattern: str,
        task: str,
        persona: str = "default",
        report_to_session: str = "",
        tomorrow: bool = False,
    ) -> dict[str, Any]:
        return await self._insert_time_or_pattern(
            label=label,
            time_or_pattern=time_or_pattern,
            tomorrow=tomorrow,
            action_type="agent_task",
            action_payload={
                "persona": persona,
                "task": task,
                "report_to_session": report_to_session,
            },
            session_id=report_to_session,
        )

    async def _tool_update_timer(
        self,
        id: str,  # noqa: A002 — matches the LLM schema
        label: str | None = None,
        fire_at_iso: str | None = None,
        pattern: str | None = None,
        active: bool | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if label is not None:
            fields["label"] = str(label)[:200]
        if fire_at_iso is not None:
            try:
                fields["fire_at"] = datetime.fromisoformat(fire_at_iso).timestamp()
            except (ValueError, TypeError) as exc:
                return {"error": f"invalid fire_at_iso: {exc}"}
        if pattern is not None:
            pattern = pattern.strip()
            if pattern == "":
                fields["repeat_pattern"] = ""
                fields["repeat_interval"] = 0
            else:
                try:
                    spec = parse_recurring(pattern)
                except ValueError as exc:
                    return {"error": f"invalid pattern: {exc}"}
                fields["repeat_pattern"] = pattern
                fields["repeat_interval"] = (
                    spec.interval_seconds if spec.is_interval else 0
                )
        if active is not None:
            fields["active"] = 1 if active else 0

        if not fields:
            return {"error": "no updatable fields provided"}

        # Whitelist columns that can be updated. This prevents SQL injection
        # if a manipulated tool call injects unexpected keys into ``fields``.
        _UPDATABLE = frozenset({
            "label", "fire_at", "repeat_pattern", "repeat_interval", "active",
        })
        safe = {k: v for k, v in fields.items() if k in _UPDATABLE}
        if not safe:
            return {"error": "no valid updatable fields after whitelist filter"}

        db = await self.api.get_db()
        sets = ", ".join(f"{col} = ?" for col in safe)
        params = list(safe.values()) + [id]
        cursor = await db.execute(
            f"UPDATE timers SET {sets} WHERE id = ? AND cancelled = 0",
            params,
        )
        await db.commit()
        if cursor.rowcount == 0:
            return {"error": f"no updatable timer with id {id!r}"}
        return {"status": "updated", "id": id, "fields": list(fields.keys())}

    async def _tool_list_timers(
        self, include_inactive: bool = False
    ) -> dict[str, Any]:
        items = await self._list_timers(include_inactive=include_inactive)
        return {
            "count": len(items),
            "timers": [
                {
                    "id": t["id"],
                    "kind": t["kind"],
                    "label": t["label"],
                    "pattern": t["repeat_pattern"] or "",
                    "active": bool(t["active"]),
                    "fires_at": datetime.fromtimestamp(
                        t["fire_at"]
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "fires_in_seconds": int(t["fire_at"] - time.time()),
                }
                for t in items
            ],
        }

    async def _tool_cancel_timer(self, id: str) -> dict[str, Any]:  # noqa: A002
        db = await self.api.get_db()
        cursor = await db.execute(
            "UPDATE timers SET cancelled = 1 WHERE id = ? "
            "AND fired = 0 AND cancelled = 0",
            (id,),
        )
        await db.commit()
        if cursor.rowcount == 0:
            return {"error": f"no active timer with id {id!r}"}
        return {"status": "cancelled", "id": id}

    # ─── WebSocket handlers ─────────────────────────────────────────

    async def _handle_ws_list(self, client: Any, message: dict[str, Any]) -> None:
        info = await self._tool_list_timers(
            include_inactive=bool(message.get("include_inactive", False))
        )
        await client.send_json({"type": "scheduler_list", **info})

    async def _handle_ws_cancel(self, client: Any, message: dict[str, Any]) -> None:
        timer_id = str(message.get("id", ""))
        if not timer_id:
            await client.send_json({"type": "error", "error": "missing id"})
            return
        result = await self._tool_cancel_timer(timer_id)
        await client.send_json({"type": "scheduler_cancelled", **result})

    async def _handle_ws_create(self, client: Any, message: dict[str, Any]) -> None:
        """Route a UI-originated create request to the right tool handler."""
        mode = str(message.get("mode", "timer"))
        label = str(message.get("label", "")).strip() or "Timer"
        try:
            if mode == "timer":
                result = await self._tool_set_timer(
                    label=label,
                    seconds=message.get("seconds"),
                    minutes=message.get("minutes"),
                )
            elif mode == "reminder":
                result = await self._tool_set_reminder(
                    time=str(message.get("time", "")),
                    label=label,
                    tomorrow=bool(message.get("tomorrow", False)),
                )
            elif mode == "recurring":
                result = await self._tool_set_recurring(
                    label=label,
                    pattern=str(message.get("pattern", "")),
                    action_type=str(message.get("action_type", "notify")),
                    action_payload=message.get("action_payload") or {},
                )
            elif mode == "proactive_chat":
                result = await self._tool_schedule_proactive(
                    label=label,
                    time_or_pattern=str(message.get("time_or_pattern", "")),
                    message=str(message.get("message", "")),
                    session_id=str(message.get("session_id", "")),
                    tomorrow=bool(message.get("tomorrow", False)),
                )
            elif mode == "agent_task":
                result = await self._tool_schedule_agent_task(
                    label=label,
                    time_or_pattern=str(message.get("time_or_pattern", "")),
                    persona=str(message.get("persona", "default")),
                    task=str(message.get("task", "")),
                    report_to_session=str(message.get("report_to_session", "")),
                    tomorrow=bool(message.get("tomorrow", False)),
                )
            else:
                result = {"error": f"unknown mode {mode!r}"}
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
        await client.send_json({"type": "scheduler_created", "mode": mode, **result})

    async def _handle_ws_update(self, client: Any, message: dict[str, Any]) -> None:
        timer_id = str(message.get("id", ""))
        if not timer_id:
            await client.send_json({"type": "error", "error": "missing id"})
            return
        result = await self._tool_update_timer(
            id=timer_id,
            label=message.get("label"),
            fire_at_iso=message.get("fire_at_iso"),
            pattern=message.get("pattern"),
            active=message.get("active"),
        )
        await client.send_json({"type": "scheduler_updated", **result})

    # ─── Internals ──────────────────────────────────────────────────

    async def _insert_time_or_pattern(
        self,
        *,
        label: str,
        time_or_pattern: str,
        tomorrow: bool,
        action_type: str,
        action_payload: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """Shared helper — parse ``time_or_pattern`` as HH:MM or as a
        recurring pattern and insert the appropriate timer row."""
        stripped = (time_or_pattern or "").strip()
        if not stripped:
            return {"error": "time_or_pattern must not be empty"}

        # Heuristic: "HH:MM" (5 chars, one colon) is a one-shot reminder;
        # anything else gets handed to the recurring parser.
        is_hhmm = (
            len(stripped) <= 5 and stripped.count(":") == 1 and " " not in stripped
        )

        timer_id = uuid.uuid4().hex[:8]
        action_blob = _encode_action(action_type, action_payload)

        if is_hhmm:
            try:
                fire_at = _parse_hhmm_to_timestamp(stripped, tomorrow)
            except ValueError as exc:
                return {"error": str(exc)}
            await self._insert_timer(
                timer_id=timer_id,
                kind=action_type,  # "proactive_chat" | "agent_task"
                label=label,
                fire_at=fire_at,
                action=action_blob,
                session_id=session_id,
            )
            return {
                "id": timer_id,
                "label": label,
                "action_type": action_type,
                "fires_at": datetime.fromtimestamp(fire_at).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }

        try:
            spec = parse_recurring(stripped)
        except ValueError as exc:
            return {"error": f"invalid pattern: {exc}"}
        fire_at = next_fire_at(spec, datetime.now()).timestamp()
        await self._insert_timer(
            timer_id=timer_id,
            kind="recurring",
            label=label,
            fire_at=fire_at,
            repeat_pattern=stripped,
            repeat_interval=spec.interval_seconds if spec.is_interval else 0,
            action=action_blob,
            session_id=session_id,
        )
        return {
            "id": timer_id,
            "label": label,
            "pattern": stripped,
            "action_type": action_type,
            "next_fire_at": datetime.fromtimestamp(fire_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

    async def _insert_timer(
        self,
        *,
        timer_id: str,
        kind: str,
        label: str,
        fire_at: float,
        repeat_pattern: str = "",
        repeat_interval: int = 0,
        action: str = "",
        session_id: str = "",
        project_id: str = "default",
    ) -> None:
        db = await self.api.get_db()
        await db.execute(
            """
            INSERT INTO timers (
                id, kind, label, fire_at, created_at,
                fired, cancelled, session_id,
                repeat_pattern, repeat_interval, action,
                project_id, last_fired_at, active
            )
            VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, 0, 1)
            """,
            (
                timer_id, kind, label, fire_at, time.time(),
                session_id,
                repeat_pattern, repeat_interval, action,
                project_id,
            ),
        )
        await db.commit()
        log.info(
            "scheduler.created",
            id=timer_id,
            kind=kind,
            label=label,
            fire_at=fire_at,
            repeat_pattern=repeat_pattern,
            action=action[:60] if action else "",
        )
        await self.api.ws_broadcast({
            "type": "scheduler_created",
            "id": timer_id,
            "kind": kind,
            "label": label,
            "fires_at": datetime.fromtimestamp(fire_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "pattern": repeat_pattern,
        })

    async def _list_timers(
        self, include_inactive: bool = False
    ) -> list[dict[str, Any]]:
        db = await self.api.get_db()
        if include_inactive:
            where = "WHERE cancelled = 0"
        else:
            where = "WHERE fired = 0 AND cancelled = 0 AND active = 1"
        cursor = await db.execute(
            f"SELECT id, kind, label, fire_at, created_at, session_id, "
            f"repeat_pattern, repeat_interval, action, project_id, "
            f"last_fired_at, active, fired, cancelled "
            f"FROM timers {where} ORDER BY fire_at ASC LIMIT ?",
            (self._max_active,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "id": row[0],
                "kind": row[1],
                "label": row[2],
                "fire_at": row[3],
                "created_at": row[4],
                "session_id": row[5] or "",
                "repeat_pattern": row[6] or "",
                "repeat_interval": int(row[7] or 0),
                "action": row[8] or "",
                "project_id": row[9] or "default",
                "last_fired_at": row[10] or 0.0,
                "active": bool(row[11]),
                "fired": bool(row[12]),
                "cancelled": bool(row[13]),
            }
            for row in rows
        ]

    async def _loop(self) -> None:
        """Background loop: check + fire due timers, emit impulses."""
        try:
            while self._running:
                await asyncio.sleep(self._check_interval)
                if not self._running:
                    break
                try:
                    await self._tick()
                except Exception as exc:  # noqa: BLE001
                    log.error("scheduler.tick_failed", error=str(exc))
        except asyncio.CancelledError:
            pass

    async def _tick(self) -> None:
        now = time.time()
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT id, kind, label, fire_at, session_id, "
            "repeat_pattern, repeat_interval, action, project_id "
            "FROM timers "
            "WHERE fired = 0 AND cancelled = 0 AND active = 1 "
            "AND fire_at <= ? ORDER BY fire_at ASC",
            (now,),
        )
        due = await cursor.fetchall()
        await cursor.close()

        for row in due:
            (
                timer_id,
                kind,
                label,
                fire_at,
                session_id,
                repeat_pattern,
                repeat_interval,
                action,
                project_id,
            ) = row
            await self._fire_timer(
                timer_id=timer_id,
                kind=kind,
                label=label,
                fire_at=float(fire_at),
                session_id=session_id or "",
                action=action or "",
                project_id=project_id or "default",
            )
            # Reschedule or mark fired.
            if repeat_pattern:
                try:
                    spec = parse_recurring(repeat_pattern)
                    next_dt = next_fire_at(spec, datetime.now())
                    next_ts = next_dt.timestamp()
                    await db.execute(
                        "UPDATE timers SET fired = 0, last_fired_at = ?, "
                        "fire_at = ? WHERE id = ?",
                        (now, next_ts, timer_id),
                    )
                except ValueError as exc:
                    log.error(
                        "scheduler.reschedule_failed",
                        id=timer_id,
                        pattern=repeat_pattern,
                        error=str(exc),
                    )
                    # Mark fired so we stop retrying a broken pattern.
                    await db.execute(
                        "UPDATE timers SET fired = 1, last_fired_at = ? "
                        "WHERE id = ?",
                        (now, timer_id),
                    )
            elif repeat_interval and repeat_interval > 0:
                await db.execute(
                    "UPDATE timers SET fired = 0, last_fired_at = ?, "
                    "fire_at = ? WHERE id = ?",
                    (now, now + repeat_interval, timer_id),
                )
            else:
                await db.execute(
                    "UPDATE timers SET fired = 1, last_fired_at = ? "
                    "WHERE id = ?",
                    (now, timer_id),
                )
            await db.commit()

        # Impulse check
        if (
            self._enable_impulses
            and self._next_impulse_at is not None
            and now >= self._next_impulse_at
        ):
            await self._fire_impulse()
            self._next_impulse_at = self._pick_next_impulse()

    async def _fire_timer(
        self,
        timer_id: str,
        kind: str,
        label: str,
        fire_at: float,
        session_id: str,
        action: str = "",
        project_id: str = "default",
    ) -> None:
        fired_at = datetime.fromtimestamp(fire_at).strftime("%H:%M")
        log.info(
            "scheduler.fired",
            id=timer_id,
            kind=kind,
            label=label,
            session=session_id,
            project=project_id,
        )

        # Broadcast to all WS clients so the GUI shows a toast
        await self.api.ws_broadcast(
            {
                "type": "scheduler_triggered",
                "id": timer_id,
                "kind": kind,
                "label": label,
                "fired_at": fired_at,
                "session_id": session_id,
                "project_id": project_id,
            }
        )

        # Dispatch structured action.
        action_data = _decode_action(action)
        action_type = action_data.get("type", "")

        # Emit event for other plugins. Includes the action type so
        # listeners (e.g. character_chat) can filter efficiently, plus the
        # raw action payload so they don't have to re-decode it.
        await self.api.emit(
            "core.scheduler_triggered",
            {
                "id": timer_id,
                "kind": kind,
                "label": label,
                "fired_at": fired_at,
                "session_id": session_id,
                "project_id": project_id,
                "action_type": action_type,
                "action": action_data,
            },
        )
        if action_type == "proactive_chat":
            await self._fire_proactive_chat(
                action_data=action_data,
                label=label,
                default_session_id=session_id,
            )
        elif action_type == "agent_task":
            await self._fire_agent_task(
                action_data=action_data,
                label=label,
                default_session_id=session_id,
            )
        elif action_type == "tool":
            await self._fire_tool_action(action_data=action_data, label=label)

    async def _fire_proactive_chat(
        self,
        *,
        action_data: dict[str, Any],
        label: str,
        default_session_id: str,
    ) -> None:
        """Nudge the agent to speak unprompted in a specific session."""
        session_id = (
            action_data.get("session_id")
            or default_session_id
            or "default"
        )
        prompt = action_data.get("prompt") or f"Erinnere an: {label}"

        # Primary path: let the agent handle it if the method is wired up.
        ok = await self.api.agent_proactive(
            session_id=session_id, prompt=prompt, label=label
        )
        if ok:
            return

        # Fallback: just emit the event so anyone listening can handle it.
        await self.api.emit(
            "core.scheduler_proactive",
            {
                "session_id": session_id,
                "text": prompt,
                "label": label,
                "from": "scheduler",
                "internal": True,
            },
        )

    async def _fire_agent_task(
        self,
        *,
        action_data: dict[str, Any],
        label: str,
        default_session_id: str,
    ) -> None:
        """Spawn an AutoAgent via the skill_writer plugin's AgentManager."""
        persona = action_data.get("persona") or "default"
        task = action_data.get("task") or label
        report_to = action_data.get("report_to_session") or default_session_id

        plugin = self.api.get_plugin("skill_writer")
        agent_manager = getattr(plugin, "_agent_manager", None) if plugin else None
        if agent_manager is None:
            log.warning(
                "scheduler.no_agent_manager",
                hint="skill_writer plugin not loaded or not enabled",
            )
            await self.api.ws_broadcast({
                "type": "agent_task_skipped",
                "label": label,
                "reason": "agent_manager unavailable",
            })
            return

        system_prompt = (
            f"Du bist ein autonomer Sub-Agent von Lexy ('{persona}'). "
            f"Dein Auftrag wurde vom Scheduler gestartet. "
            f"Arbeite präzise und fasse das Ergebnis am Ende in wenigen "
            f"Sätzen auf Deutsch zusammen."
        )
        result = await agent_manager.spawn(
            name=f"sched_{persona}",
            task=task,
            system_prompt=system_prompt,
            brain="e4b",
        )

        # spawn() returns {agent_id, name, task, status} or {error}
        await self.api.ws_broadcast({
            "type": "agent_task_spawned",
            "label": label,
            "session_id": report_to,
            "persona": persona,
            **result,
        })

    async def _fire_tool_action(
        self,
        *,
        action_data: dict[str, Any],
        label: str,
    ) -> None:
        """Execute a whitelisted tool via the plugin tool registry."""
        tool_name = str(action_data.get("tool") or "")
        if not tool_name:
            log.warning("scheduler.tool_action_missing_name", label=label)
            return
        args = action_data.get("args") or {}
        registry = self.api.get_tool_registry()
        if registry is None:
            log.warning("scheduler.no_tool_registry")
            return
        try:
            result = await registry.execute(tool_name, dict(args))
            await self.api.ws_broadcast({
                "type": "scheduler_tool_fired",
                "label": label,
                "tool": tool_name,
                "ok": bool(getattr(result, "success", False)),
            })
        except Exception as exc:  # noqa: BLE001
            log.error(
                "scheduler.tool_action_failed",
                tool=tool_name,
                error=str(exc),
            )

    async def _fire_impulse(self) -> None:
        """Kick off a daydream impulse via the chat agent."""
        situations = [
            "Du hörst draußen einen Vogel zwitschern und musst kurz daran denken.",
            "Dir fällt gerade etwas ein, was du Mike erzählen wolltest.",
            "Du merkst dass es schon länger still ist und fragst dich was Mike gerade macht.",
            "Ein Gedanke über etwas das gestern besprochen wurde poppt auf.",
        ]
        situation = random.choice(situations)
        prompt = (
            f"Impuls: {situation}\n\n"
            "Reagiere kurz und natürlich in-character. Schreibe nur 1-2 Sätze."
        )
        log.info("scheduler.impulse", situation=situation)
        await self.api.emit(
            "core.scheduler_impulse",
            {"situation": situation, "prompt": prompt},
        )

    def _pick_next_impulse(self) -> float:
        """Schedule the next impulse 45-120 minutes from now, within active hours."""
        delay = random.randint(45 * 60, 120 * 60)
        next_time = datetime.fromtimestamp(time.time() + delay)
        # Clamp into the active window
        if next_time.hour < self._impulse_min_hour:
            next_time = next_time.replace(
                hour=self._impulse_min_hour, minute=random.randint(0, 59)
            )
        elif next_time.hour >= self._impulse_max_hour:
            next_time = next_time.replace(
                hour=self._impulse_min_hour,
                minute=random.randint(0, 59),
            ) + timedelta(days=1)
        return next_time.timestamp()


# ─── Module helpers ─────────────────────────────────────────────────────────


def _parse_hhmm_to_timestamp(time_str: str, tomorrow: bool) -> float:
    """Parse HH:MM → absolute unix timestamp today (or tomorrow if past)."""
    try:
        hour_str, minute_str = time_str.split(":")
        hour = int(hour_str)
        minute = int(minute_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"invalid time format: {time_str!r} (expected HH:MM)"
        ) from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid time range: {time_str!r}")
    now = datetime.now()
    fire_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if tomorrow or fire_time <= now:
        fire_time = fire_time + timedelta(days=1)
    return fire_time.timestamp()


def _encode_action(action_type: str, payload: dict[str, Any]) -> str:
    """Encode an action as a compact JSON blob for the DB column.

    Returns an empty string for the bare 'notify' action so the legacy
    code path (no action) still applies cleanly.
    """
    if action_type in ("", "notify"):
        return ""
    return json.dumps({"type": action_type, **payload}, ensure_ascii=False)


def _decode_action(action: str) -> dict[str, Any]:
    """Decode the DB action blob back into a dict. Robust to empties/garbage."""
    if not action:
        return {}
    try:
        data = json.loads(action)
        if not isinstance(data, dict):
            return {}
        return data
    except (ValueError, TypeError):
        return {}
