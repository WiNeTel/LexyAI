"""
Lexy AI - Dashboard Plugin.

Smart-Mirror-style widget dashboard with a configurable grid layout.

Features:

* **Widget Registry** -- any plugin can register dashboard widgets via
  ``DashboardPlugin.register_widget()`` or by implementing ``BaseWidget``.
* **Built-in widgets** -- clock, weather, memory_stats, system_status,
  sessions, thoughts, search, notes (see ``widgets/`` subpackage).
* **Persistent layouts** -- per-user grid arrangement stored in SQLite.
* **Persistent notes** -- sticky notes with color and position.
* **Background refresh** -- periodically polls each widget's ``get_data()``
  and broadcasts only changed payloads to connected clients.
* **WS handlers** -- frontend communicates via typed WebSocket messages
  (get/save layout, widget data, note CRUD, search).
* **LLM tool** -- ``get_dashboard_summary`` lets the LLM describe
  the current dashboard state in natural language.

Architecture:

    Frontend  --WS-->  DashboardPlugin  --calls-->  WidgetRegistry
                                          |
                                          +-->  SQLite (layouts, notes)
                                          +-->  EventBus (autonomous_thought)
                                          +-->  PluginAPI (memory, plugins)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from plugins.dashboard.widgets import ALL_WIDGET_CLASSES, BaseWidget
from plugins.dashboard.widgets.notes_widget import NotesWidget
from plugins.dashboard.widgets.search_widget import SearchWidget
from plugins.dashboard.widgets.thoughts_widget import ThoughtsWidget

log = get_logger(module="dashboard_plugin")


# ─── Widget Registration ────────────────────────────────────────────────────

@dataclass
class WidgetRegistration:
    """Metadata for a registered dashboard widget."""

    id: str
    data_fn: Callable[[], Awaitable[dict[str, Any]]]
    refresh_interval: float
    default_size: tuple[int, int]
    title: str
    source: str  # Plugin name that registered this widget


# ─── Tool Schema ────────────────────────────────────────────────────────────

DASHBOARD_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "widgets": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional list of widget IDs to include in the summary. "
                "If empty or omitted, all widgets are included."
            ),
        },
    },
}


# ─── Plugin ─────────────────────────────────────────────────────────────────

class DashboardPlugin(BasePlugin):
    """Configurable dashboard with plugin-provided widgets."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)

        # Widget registry
        self._widgets: dict[str, WidgetRegistration] = {}
        self._widget_instances: dict[str, BaseWidget] = {}
        self._widget_data_cache: dict[str, dict[str, Any]] = {}

        # Config
        self._refresh_interval: float = 30.0  # fallback if a widget has no override
        self._default_layout: list[dict[str, Any]] = []
        # Per-widget refresh override map from config (widget_id -> seconds).
        # A value of 0 disables the widget's periodic refresh (push-only).
        self._widget_intervals: dict[str, float] = {}

        # Background refresh — one asyncio task per widget.
        self._running: bool = False
        self._widget_tasks: dict[str, asyncio.Task[None]] = {}

    # ─── Lifecycle ──────────────────────────────────────────────────────

    async def on_load(self) -> None:
        """Initialise SQLite tables, load config, instantiate built-in widgets."""
        config = self.api.get_config()
        self._refresh_interval = float(config.get("refresh_interval_seconds", 30))
        self._default_layout = list(config.get("default_layout", []))
        self._widget_intervals = self._parse_interval_overrides(
            config.get("widget_intervals", {})
        )

        # Create SQLite tables for layouts and notes
        db = await self.api.get_db()
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS layouts (
                user_id TEXT PRIMARY KEY,
                layout_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                color TEXT NOT NULL DEFAULT '#fef08a',
                position_index INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        await db.commit()

        # Instantiate and register all built-in widgets
        for widget_cls in ALL_WIDGET_CLASSES:
            try:
                instance = widget_cls(self.api)
                self._widget_instances[instance.widget_id] = instance
                self._register_widget_from_instance(instance, source="dashboard")
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "dashboard.widget_init_failed",
                    widget=getattr(widget_cls, "widget_id", "?"),
                    error=str(exc),
                )

        log.info(
            "dashboard.loaded",
            widgets=list(self._widgets.keys()),
            refresh_interval=self._refresh_interval,
        )

    async def on_enable(self) -> None:
        """Register WS handlers, tools, event subscriptions, and start refresh loop."""

        # ─── WS Handlers ──────────────────────────────────────────
        self.api.register_ws_handler(
            "get_dashboard_widgets", self._handle_get_widgets
        )
        self.api.register_ws_handler(
            "get_dashboard_layout", self._handle_get_layout
        )
        self.api.register_ws_handler(
            "save_dashboard_layout", self._handle_save_layout
        )
        self.api.register_ws_handler(
            "dashboard_note_save", self._handle_note_save
        )
        self.api.register_ws_handler(
            "dashboard_note_delete", self._handle_note_delete
        )
        self.api.register_ws_handler(
            "dashboard_search", self._handle_search
        )

        # ─── LLM Tool ─────────────────────────────────────────────
        self.api.register_tool(
            name="get_dashboard_summary",
            handler=self._tool_dashboard_summary,
            description=(
                "Get a summary of the current dashboard state including "
                "clock, weather, system status, memory stats, active "
                "sessions, and recent thoughts. Useful for answering "
                "questions like 'wie sieht mein Dashboard aus?' or "
                "'was zeigt das Dashboard gerade an?'."
            ),
            schema=DASHBOARD_SUMMARY_SCHEMA,
        )

        # ─── Event Subscriptions ──────────────────────────────────
        # Subscribe the ThoughtsWidget to autonomous_thought events
        thoughts_widget = self._widget_instances.get("thoughts")
        if isinstance(thoughts_widget, ThoughtsWidget):
            thoughts_widget.subscribe()

        # Listen for system events to push updates
        self.api.on_event("core.system_ready", self._on_system_ready)
        # Push-only widgets: re-broadcast whenever the underlying event fires
        # so connected dashboards update instantly without waiting for a poll.
        self.api.on_event("core.autonomous_thought", self._on_autonomous_thought)

        # ─── Per-Widget Background Refresh Tasks ──────────────────
        self._running = True
        self._widget_tasks = {}
        for widget_id, reg in self._widgets.items():
            if reg.refresh_interval <= 0:
                # Push-only widget — no periodic poll.
                continue
            self._widget_tasks[widget_id] = asyncio.create_task(
                self._widget_refresh_task(widget_id, reg),
                name=f"dashboard.refresh.{widget_id}",
            )

        log.info(
            "dashboard.enabled",
            widget_count=len(self._widgets),
            periodic_widgets=list(self._widget_tasks.keys()),
            intervals={wid: reg.refresh_interval for wid, reg in self._widgets.items()},
        )

    async def on_disable(self) -> None:
        """Stop per-widget refresh tasks and clean up."""
        self._running = False
        tasks = list(self._widget_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._widget_tasks.clear()
        self._widget_data_cache.clear()
        log.info("dashboard.disabled")

    # ─── Config helpers ────────────────────────────────────────────────

    @staticmethod
    def _parse_interval_overrides(
        raw: Any,
    ) -> dict[str, float]:
        """Normalise the ``widget_intervals`` config dict into {id: seconds}.

        Accepts numeric values only; silently drops anything invalid so one
        bad entry can't break the whole plugin.
        """
        out: dict[str, float] = {}
        if not isinstance(raw, dict):
            return out
        for key, value in raw.items():
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                log.warning(
                    "dashboard.widget_interval_invalid",
                    widget_id=key,
                    value=value,
                )
                continue
            if seconds < 0:
                seconds = 0.0
            out[str(key)] = seconds
        return out

    # ─── Public API: Widget Registration ────────────────────────────────

    def register_widget(
        self,
        widget_id: str,
        data_fn: Callable[[], Awaitable[dict[str, Any]]],
        refresh_interval: float = 30.0,
        default_size: tuple[int, int] = (1, 1),
        title: str = "",
        source: str = "unknown",
    ) -> None:
        """
        Register an external widget. Other plugins call this to add
        their own widgets to the dashboard.

        Args:
            widget_id: Unique slug for the widget.
            data_fn: Async callable returning the widget's current data dict.
            refresh_interval: Seconds between auto-refresh calls (0 = no auto).
            default_size: (width, height) grid units.
            title: Human-readable display name.
            source: Name of the plugin registering this widget.
        """
        if widget_id in self._widgets:
            log.warning(
                "dashboard.widget_already_registered",
                widget_id=widget_id,
                existing_source=self._widgets[widget_id].source,
                new_source=source,
            )
            return

        # Config override wins over the widget's own default. Value 0
        # explicitly switches a widget into push-only mode.
        effective_interval = refresh_interval
        if widget_id in self._widget_intervals:
            effective_interval = self._widget_intervals[widget_id]

        reg = WidgetRegistration(
            id=widget_id,
            data_fn=data_fn,
            refresh_interval=effective_interval,
            default_size=default_size,
            title=title or widget_id,
            source=source,
        )
        self._widgets[widget_id] = reg
        log.info(
            "dashboard.widget_registered",
            widget_id=widget_id,
            source=source,
            refresh=effective_interval,
            default_refresh=refresh_interval,
        )

    def _register_widget_from_instance(
        self, widget: BaseWidget, source: str
    ) -> None:
        """Register a BaseWidget instance into the WidgetRegistration map."""
        self.register_widget(
            widget_id=widget.widget_id,
            data_fn=widget.get_data,
            refresh_interval=widget.refresh_interval,
            default_size=widget.default_size,
            title=widget.title,
            source=source,
        )

    # ─── WS Handlers ───────────────────────────────────────────────────

    async def _handle_get_widgets(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Fetch all widget data and send to client."""
        widget_data = await self._fetch_all_widget_data()

        # Build widget manifest (metadata + data)
        widgets_payload: list[dict[str, Any]] = []
        for widget_id, reg in self._widgets.items():
            widgets_payload.append({
                "widget_id": reg.id,
                "title": reg.title,
                "default_size": list(reg.default_size),
                "refresh_interval": reg.refresh_interval,
                "source": reg.source,
                "data": widget_data.get(widget_id, {}),
            })

        await client.send_json({
            "type": "dashboard_widgets",
            "widgets": widgets_payload,
            "count": len(widgets_payload),
        })

    async def _handle_get_layout(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Read layout from SQLite for the requesting user."""
        user_id = str(message.get("user_id", "default"))
        layout = await self._load_layout(user_id)

        await client.send_json({
            "type": "dashboard_layout",
            "user_id": user_id,
            "layout": layout,
        })

    async def _handle_save_layout(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Write layout to SQLite."""
        user_id = str(message.get("user_id", "default"))
        layout = message.get("layout", [])

        if not isinstance(layout, list):
            await client.send_json({
                "type": "error",
                "error": "layout must be a list of widget placement objects",
            })
            return

        await self._save_layout(user_id, layout)

        await client.send_json({
            "type": "dashboard_layout_saved",
            "user_id": user_id,
            "layout": layout,
        })
        log.info("dashboard.layout_saved", user_id=user_id, items=len(layout))

    async def _handle_note_save(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Create or update a sticky note."""
        notes_widget = self._widget_instances.get("notes")
        if not isinstance(notes_widget, NotesWidget):
            await client.send_json({
                "type": "error",
                "error": "notes widget not available",
            })
            return

        db = await self.api.get_db()
        note = await notes_widget.save_note(
            db=db,
            note_id=message.get("note_id") or message.get("id"),
            title=str(message.get("title", "")),
            content=str(message.get("content", "")),
            color=str(message.get("color", "#fef08a")),
            position_index=int(message.get("position_index", 0)),
        )

        await client.send_json({
            "type": "dashboard_note_saved",
            "note": note,
        })

        # Broadcast note update to all clients so other tabs stay in sync
        await self.api.ws_broadcast({
            "type": "dashboard_widget_update",
            "widget_id": "notes",
            "data": await notes_widget.get_data(),
        })

    async def _handle_note_delete(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Delete a sticky note by id."""
        note_id = str(message.get("note_id") or message.get("id", ""))
        if not note_id:
            await client.send_json({
                "type": "error",
                "error": "missing note_id",
            })
            return

        notes_widget = self._widget_instances.get("notes")
        if not isinstance(notes_widget, NotesWidget):
            await client.send_json({
                "type": "error",
                "error": "notes widget not available",
            })
            return

        db = await self.api.get_db()
        deleted = await notes_widget.delete_note(db=db, note_id=note_id)

        await client.send_json({
            "type": "dashboard_note_deleted",
            "note_id": note_id,
            "deleted": deleted,
        })

        # Broadcast updated notes list
        if deleted:
            await self.api.ws_broadcast({
                "type": "dashboard_widget_update",
                "widget_id": "notes",
                "data": await notes_widget.get_data(),
            })

    async def _handle_search(
        self, client: Any, message: dict[str, Any]
    ) -> None:
        """Execute a memory search from the dashboard."""
        search_widget = self._widget_instances.get("search")
        if not isinstance(search_widget, SearchWidget):
            await client.send_json({
                "type": "error",
                "error": "search widget not available",
            })
            return

        query = str(message.get("query", ""))
        collection = message.get("collection")
        limit = int(message.get("limit", 5))

        results = await search_widget.search(
            query=query,
            collection=collection,
            limit=limit,
        )

        await client.send_json({
            "type": "dashboard_search_results",
            **results,
        })

    # ─── LLM Tool ──────────────────────────────────────────────────────

    async def _tool_dashboard_summary(
        self,
        widgets: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a machine-readable summary of current dashboard state."""
        all_data = await self._fetch_all_widget_data()

        if widgets:
            # Filter to requested widgets only
            filtered: dict[str, Any] = {
                wid: data for wid, data in all_data.items() if wid in widgets
            }
        else:
            filtered = all_data

        summary: dict[str, Any] = {
            "widget_count": len(self._widgets),
            "registered_widgets": [
                {
                    "id": reg.id,
                    "title": reg.title,
                    "source": reg.source,
                }
                for reg in self._widgets.values()
            ],
            "widget_data": filtered,
        }
        return summary

    # ─── Event Handlers ────────────────────────────────────────────────

    async def _on_system_ready(self, data: dict[str, Any]) -> None:
        """On system ready, do an initial fetch of all widgets."""
        log.debug("dashboard.system_ready_refresh")
        try:
            await self._fetch_all_widget_data()
        except Exception as exc:  # noqa: BLE001
            log.warning("dashboard.initial_fetch_failed", error=str(exc))

    async def _on_autonomous_thought(self, event: Any) -> None:
        """
        Push-refresh the thoughts widget whenever a new autonomous thought
        is emitted. The ThoughtsWidget's own subscription updates its cache
        first (EventBus fan-out order is registration order, and the widget
        subscribes in on_enable before this handler), so ``get_data`` here
        returns the freshly-appended entry.
        """
        try:
            await self._broadcast_widget("thoughts")
        except Exception as exc:  # noqa: BLE001
            log.warning("dashboard.thoughts_push_failed", error=str(exc))

    # ─── Background Refresh (per widget) ───────────────────────────────

    async def _widget_refresh_task(
        self, widget_id: str, reg: WidgetRegistration
    ) -> None:
        """
        Periodically poll a single widget's ``data_fn`` at its own interval.

        Unlike the previous single-loop design, each widget runs on its own
        schedule (weather every 30 min, sessions every 30 s, etc.). We always
        broadcast the fresh snapshot on every tick — the cost is negligible
        over a local WS, and it guarantees reconnecting clients get current
        data instead of waiting for the next diff.
        """
        try:
            while self._running:
                await asyncio.sleep(reg.refresh_interval)
                if not self._running:
                    break
                try:
                    new_data = await asyncio.wait_for(
                        reg.data_fn(), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "dashboard.widget_timeout",
                        widget_id=widget_id,
                        timeout=15.0,
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "dashboard.widget_fetch_failed",
                        widget_id=widget_id,
                        error=str(exc),
                    )
                    continue

                self._widget_data_cache[widget_id] = new_data
                try:
                    await self.api.ws_broadcast({
                        "type": "dashboard_widget_update",
                        "widget_id": widget_id,
                        "data": new_data,
                    })
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "dashboard.widget_broadcast_failed",
                        widget_id=widget_id,
                        error=str(exc),
                    )
        except asyncio.CancelledError:
            pass

    async def _broadcast_widget(self, widget_id: str) -> None:
        """Fetch one widget's current data and broadcast it immediately.

        Used by event handlers (e.g. autonomous_thought → refresh thoughts
        widget) so push-only widgets can still appear live on the dashboard.
        """
        reg = self._widgets.get(widget_id)
        if reg is None:
            return
        try:
            new_data = await asyncio.wait_for(reg.data_fn(), timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "dashboard.widget_broadcast_fetch_failed",
                widget_id=widget_id,
                error=str(exc),
            )
            return
        self._widget_data_cache[widget_id] = new_data
        try:
            await self.api.ws_broadcast({
                "type": "dashboard_widget_update",
                "widget_id": widget_id,
                "data": new_data,
            })
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "dashboard.widget_broadcast_failed",
                widget_id=widget_id,
                error=str(exc),
            )

    # ─── Data Fetching ─────────────────────────────────────────────────

    async def _fetch_all_widget_data(self) -> dict[str, dict[str, Any]]:
        """Fetch data from all registered widgets and update the cache."""
        result: dict[str, dict[str, Any]] = {}

        for widget_id, reg in self._widgets.items():
            try:
                data = await asyncio.wait_for(
                    reg.data_fn(), timeout=10.0
                )
            except asyncio.TimeoutError:
                log.warning(
                    "dashboard.widget_timeout",
                    widget_id=widget_id,
                )
                data = self._widget_data_cache.get(widget_id, {})
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "dashboard.widget_fetch_failed",
                    widget_id=widget_id,
                    error=str(exc),
                )
                data = self._widget_data_cache.get(widget_id, {})

            result[widget_id] = data
            self._widget_data_cache[widget_id] = data

        return result

    # ─── Layout Persistence ────────────────────────────────────────────

    async def _load_layout(self, user_id: str) -> list[dict[str, Any]]:
        """Load layout from SQLite, falling back to the default from config."""
        db = await self.api.get_db()
        cursor = await db.execute(
            "SELECT layout_json FROM layouts WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row is not None and row[0]:
            try:
                layout = json.loads(row[0])
                if isinstance(layout, list):
                    return layout
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning(
                    "dashboard.layout_parse_failed",
                    user_id=user_id,
                    error=str(exc),
                )

        # Return default layout from plugin config
        return list(self._default_layout)

    async def _save_layout(
        self, user_id: str, layout: list[dict[str, Any]]
    ) -> None:
        """Persist layout to SQLite."""
        db = await self.api.get_db()
        layout_json = json.dumps(layout, ensure_ascii=False)
        now = time.time()

        await db.execute(
            """
            INSERT INTO layouts (user_id, layout_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                layout_json = excluded.layout_json,
                updated_at = excluded.updated_at
            """,
            (user_id, layout_json, now),
        )
        await db.commit()
