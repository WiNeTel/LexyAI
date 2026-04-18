"""
Lexy AI - Dashboard Sessions Widget.

Shows active session count, per-session message counts, and most-recent
activity timestamps.  Reads directly from the ``SessionStore`` on
``LexyApp`` without duplicating the underlying data.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.sessions")


class SessionsWidget(BaseWidget):
    """Active chat sessions overview."""

    widget_id: str = "sessions"
    title: str = "Sessions"
    default_size: tuple[int, int] = (2, 2)
    refresh_interval: float = 15.0

    def __init__(self, api: Any) -> None:
        super().__init__(api)

    async def get_data(self) -> dict[str, Any]:
        """Snapshot of all sessions with message counts."""
        store = self._api._app.session_store
        session_ids: list[str] = store.sessions()
        total_messages: int = 0

        sessions: list[dict[str, Any]] = []
        for sid in session_ids:
            history = store.get(sid)
            msg_count = len(history)
            total_messages += msg_count

            # Derive last activity from the most recent message.
            # SessionStore entries are ``{"role": ..., "content": ...}``
            # without explicit timestamps, so we use the message index as
            # a proxy for ordering and leave the frontend to show relative
            # positioning. If the entry carried a ``time`` key we'd use it.
            last_content: str = ""
            if history:
                last_msg = history[-1]
                last_content = str(last_msg.get("content", ""))[:80]

            sessions.append({
                "id": sid,
                "messages": msg_count,
                "last_snippet": last_content,
            })

        # Sort: busiest sessions first
        sessions.sort(key=lambda s: s["messages"], reverse=True)

        return {
            "active_count": len(session_ids),
            "sessions": sessions,
            "total_messages": total_messages,
        }
