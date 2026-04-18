"""
Lexy AI - Dashboard Notes Widget.

Simple CRUD for quick notes stored in the dashboard plugin's private
aiosqlite database (managed via ``PluginAPI.get_db()``).  Notes are
lightweight text snippets the user pins to the dashboard.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.notes")


class NotesWidget(BaseWidget):
    """Persistent quick notes (SQLite-backed)."""

    widget_id: str = "notes"
    title: str = "Notizen"
    default_size: tuple[int, int] = (2, 3)
    refresh_interval: float = 0.0  # on-demand only

    def __init__(self, api: Any) -> None:
        super().__init__(api)
        self._db_ready: bool = False

    # ─── DB schema ──────────────────────────────────────────────────

    async def _ensure_table(self) -> aiosqlite.Connection:
        """
        Ensure the ``notes`` table exists and return the connection.

        Uses the dashboard plugin's shared aiosqlite connection (WAL mode
        already configured by ``PluginAPI.get_db()``).
        """
        db = await self._api.get_db()
        if not self._db_ready:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL DEFAULT '',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                )
                """
            )
            await db.commit()
            self._db_ready = True
        return db

    # ─── Widget data ────────────────────────────────────────────────

    async def get_data(self) -> dict[str, Any]:
        """Return all notes ordered by most-recently updated first."""
        try:
            db = await self._ensure_table()
            cursor = await db.execute(
                "SELECT id, content, created_at, updated_at "
                "FROM notes ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
            await cursor.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("widget.notes.load_failed", error=str(exc))
            return {"notes": [], "error": str(exc)}

        notes: list[dict[str, Any]] = [
            {
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

        return {"notes": notes}

    # ─── CRUD operations ────────────────────────────────────────────

    async def create_note(self, content: str) -> dict[str, Any]:
        """Create a new note. Returns the created note dict."""
        note_id = uuid.uuid4().hex[:12]
        now = time.time()

        try:
            db = await self._ensure_table()
            await db.execute(
                "INSERT INTO notes (id, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (note_id, content, now, now),
            )
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            log.error("widget.notes.create_failed", error=str(exc))
            return {"error": str(exc)}

        log.info("widget.notes.created", id=note_id)
        return {
            "id": note_id,
            "content": content,
            "created_at": now,
            "updated_at": now,
        }

    async def update_note(self, note_id: str, content: str) -> dict[str, Any]:
        """Update an existing note's content. Returns the updated note dict."""
        now = time.time()

        try:
            db = await self._ensure_table()
            cursor = await db.execute(
                "UPDATE notes SET content = ?, updated_at = ? WHERE id = ?",
                (content, now, note_id),
            )
            await db.commit()
            if cursor.rowcount == 0:
                return {"error": f"note not found: {note_id}"}
        except Exception as exc:  # noqa: BLE001
            log.error("widget.notes.update_failed", id=note_id, error=str(exc))
            return {"error": str(exc)}

        log.info("widget.notes.updated", id=note_id)
        return {
            "id": note_id,
            "content": content,
            "updated_at": now,
        }

    async def delete_note(self, note_id: str) -> dict[str, Any]:
        """Delete a note by id. Returns the deleted id on success."""
        try:
            db = await self._ensure_table()
            cursor = await db.execute(
                "DELETE FROM notes WHERE id = ?", (note_id,)
            )
            await db.commit()
            if cursor.rowcount == 0:
                return {"error": f"note not found: {note_id}"}
        except Exception as exc:  # noqa: BLE001
            log.error("widget.notes.delete_failed", id=note_id, error=str(exc))
            return {"error": str(exc)}

        log.info("widget.notes.deleted", id=note_id)
        return {"deleted": note_id}
