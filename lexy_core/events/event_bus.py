"""
Lexy AI - String-based EventBus.

Async publish/subscribe with wildcard event names. Core events use
the `core.` prefix; plugins use `<plugin_name>.` prefix.

Example:
    bus = EventBus()
    bus.on("core.user_message", my_handler, source="my_plugin")
    await bus.emit("core.user_message", {"text": "Hello"})
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from lexy_core.utils.logging import get_logger

log = get_logger(module="event_bus")

#: An event handler is a sync or async callable receiving an Event.
EventHandler = Callable[["Event"], Any | Awaitable[Any]]


@dataclass
class Event:
    """A single event traveling on the EventBus."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "core"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event into a JSON-friendly dict."""
        return {
            "id": self.id,
            "name": self.name,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


class EventBus:
    """
    Central event bus with wildcard subscriptions and source tracking.

    Features
    --------
    * Wildcard subscribe: ``core.*`` matches ``core.ready``, ``core.shutdown``.
    * Global wildcard ``*`` receives every event.
    * Source tracking: ``off_all("my_plugin")`` cleans up safely.
    * Bounded event history (default 500 events).
    * Sync and async callbacks supported.
    """

    def __init__(self, max_history: int = 500) -> None:
        self._listeners: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history = max_history
        # source → list of (event_name, handler) for cleanup
        self._source_listeners: dict[str, list[tuple[str, EventHandler]]] = defaultdict(list)

    # ─── Subscription ────────────────────────────────────────────────

    def on(self, event_name: str, callback: EventHandler, source: str = "core") -> None:
        """Register an event listener."""
        self._listeners[event_name].append(callback)
        self._source_listeners[source].append((event_name, callback))
        log.debug("event_bus.subscribed", event_name=event_name, source=source)

    def off(self, event_name: str, callback: EventHandler) -> bool:
        """Remove a single listener. Returns True if it was registered."""
        listeners = self._listeners.get(event_name, [])
        if callback in listeners:
            listeners.remove(callback)
            return True
        return False

    def off_all(self, source: str) -> int:
        """Remove every listener that was registered with the given source."""
        removed = 0
        for event_name, callback in self._source_listeners.get(source, []):
            handlers = self._listeners.get(event_name, [])
            if callback in handlers:
                handlers.remove(callback)
                removed += 1
        self._source_listeners.pop(source, None)
        if removed:
            log.debug("event_bus.cleaned", source=source, removed=removed)
        return removed

    # ─── Emission ────────────────────────────────────────────────────

    async def emit(
        self,
        event_name: str,
        data: dict[str, Any] | None = None,
        source: str = "core",
    ) -> int:
        """
        Emit an event to all matching listeners.

        Returns
        -------
        int
            Number of listeners notified.
        """
        event = Event(name=event_name, data=data or {}, source=source)

        # Bounded history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        callbacks: list[EventHandler] = []
        callbacks.extend(self._listeners.get(event_name, []))

        # Wildcard match: 'core.*'
        parts = event_name.split(".")
        if len(parts) >= 2:
            wildcard = f"{parts[0]}.*"
            callbacks.extend(self._listeners.get(wildcard, []))

        # Global '*'
        callbacks.extend(self._listeners.get("*", []))

        handled = 0
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
                handled += 1
            except Exception as exc:  # noqa: BLE001 — handler errors must not crash the bus
                log.error(
                    "event_bus.handler_error",
                    event_name=event_name,
                    handler=getattr(cb, "__name__", repr(cb)),
                    error=str(exc),
                )

        return handled

    # ─── Introspection ───────────────────────────────────────────────

    def get_history(
        self, event_name: str | None = None, limit: int = 50
    ) -> list[Event]:
        """Return recent events, optionally filtered by name."""
        events = self._history
        if event_name:
            events = [event for event in events if event.name == event_name]
        return events[-limit:]

    def get_listener_count(self, event_name: str | None = None) -> int:
        """Number of registered listeners (total or for one event)."""
        if event_name:
            return len(self._listeners.get(event_name, []))
        return sum(len(handlers) for handlers in self._listeners.values())
