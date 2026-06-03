"""
Lexy AI — Skill execution log + refine decision (Phase P5).

Every ``run_skill`` outcome is recorded here so a failing skill can be
self-repaired: once it has failed ``refine_after_failures`` times the plugin
drafts a patched version (live), archiving the old one. The recent error
messages stored here feed the optimizer prompt.

``should_refine`` is a pure function so the trigger logic is unit-testable
without a database.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

log = get_logger(module="skill_executions")


def should_refine(failure_count: int, *, threshold: int) -> bool:
    """True once a skill has accumulated enough failures to warrant a refine."""
    return threshold > 0 and failure_count >= threshold


class SkillExecutionLog:
    """SQLite-backed log of skill runs (shared plugin DB connection)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def init_tables(self) -> None:
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_executions (
                id          TEXT PRIMARY KEY,
                skill_name  TEXT NOT NULL,
                ok          INTEGER NOT NULL,
                error       TEXT NOT NULL DEFAULT '',
                args_json   TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_exec_name "
            "ON skill_executions(skill_name, created_at)"
        )
        await self._db.commit()

    async def record(
        self,
        skill_name: str,
        *,
        ok: bool,
        error: str = "",
        args_json: str = "",
    ) -> None:
        await self._db.execute(
            "INSERT INTO skill_executions (id, skill_name, ok, error, args_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex[:12],
                skill_name,
                1 if ok else 0,
                error[:1000],
                args_json[:1000],
                time.time(),
            ),
        )
        await self._db.commit()

    async def recent_failures(
        self, skill_name: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            "SELECT error, args_json, created_at FROM skill_executions "
            "WHERE skill_name = ? AND ok = 0 ORDER BY created_at DESC LIMIT ?",
            (skill_name, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {"error": row[0], "args_json": row[1], "created_at": row[2]}
            for row in rows
        ]
