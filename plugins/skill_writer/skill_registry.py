"""
Lexy AI - Skill Registry.

SQLite-backed registry that tracks all installed skills, their metadata,
and usage statistics. Also scans the skills directory on disk to
auto-register skill files that were added manually.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

from .skill_template import parse_skill_header

log = get_logger(module="skill_registry")


@dataclass
class SkillEntry:
    """One registered skill with its metadata and stats."""

    id: str
    name: str
    description: str
    file_path: str
    status: str  # active, disabled, failed
    created_at: float
    updated_at: float | None
    usage_count: int
    success_count: int
    failure_count: int
    last_used_at: float | None
    source: str  # auto, manual, sub_agent


class SkillRegistry:
    """
    Manages the lifecycle of skill files.

    Backed by an aiosqlite connection (shared with the plugin's DB).
    Supports registration, lookup, stats tracking, and disk scanning.
    """

    def __init__(self, db: aiosqlite.Connection, skills_path: Path) -> None:
        self._db = db
        self._skills_path = skills_path

    async def init_tables(self) -> None:
        """Create the skills table if it doesn't exist."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                file_path   TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                created_at  REAL NOT NULL,
                updated_at  REAL,
                usage_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_used_at REAL,
                source      TEXT NOT NULL DEFAULT 'manual'
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status)"
        )
        await self._db.commit()
        log.info("skill_registry.tables_ready")

    async def scan_disk(self) -> int:
        """
        Scan skills_path for .py files, register any not already in DB.

        Parses the docstring header of each file to extract metadata.

        Returns:
            Number of newly registered skills.
        """
        if not self._skills_path.exists():
            return 0

        count = 0
        for py_file in self._skills_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            skill_name = py_file.stem
            existing = await self.get(skill_name)
            if existing is not None:
                continue

            # Header parsen fuer Beschreibung
            try:
                source = py_file.read_text(encoding="utf-8")
                header = parse_skill_header(source)
                description = header.get("description", "")
                author = header.get("author", "manual")
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "skill_registry.scan_read_error",
                    file=str(py_file),
                    error=str(exc),
                )
                continue

            source_tag = "manual" if author == "manual" else author
            await self.register(
                name=skill_name,
                description=description or f"Skill from {py_file.name}",
                file_path=str(py_file.resolve()),
                source=source_tag,
            )
            count += 1
            log.info("skill_registry.scanned", name=skill_name, file=str(py_file))

        return count

    async def register(
        self,
        name: str,
        description: str,
        file_path: str,
        source: str = "manual",
    ) -> str:
        """
        Register a new skill.

        Args:
            name:        Unique skill name (snake_case).
            description: One-line description.
            file_path:   Absolute path to the .py skill file.
            source:      Origin: ``auto``, ``manual``, or ``sub_agent``.

        Returns:
            The generated skill ID.
        """
        skill_id = uuid.uuid4().hex[:12]
        now = time.time()

        await self._db.execute(
            """
            INSERT INTO skills
                (id, name, description, file_path, status, created_at, source)
            VALUES (?, ?, ?, ?, 'active', ?, ?)
            """,
            (skill_id, name, description, file_path, now, source),
        )
        await self._db.commit()
        log.info(
            "skill_registry.registered",
            id=skill_id,
            name=name,
            source=source,
        )
        return skill_id

    async def get(self, name: str) -> SkillEntry | None:
        """Look up a skill by name. Returns None if not found."""
        cursor = await self._db.execute(
            """
            SELECT id, name, description, file_path, status,
                   created_at, updated_at, usage_count, success_count,
                   failure_count, last_used_at, source
            FROM skills WHERE name = ?
            """,
            (name,),
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row is None:
            return None

        return SkillEntry(
            id=row[0],
            name=row[1],
            description=row[2],
            file_path=row[3],
            status=row[4],
            created_at=row[5],
            updated_at=row[6],
            usage_count=row[7],
            success_count=row[8],
            failure_count=row[9],
            last_used_at=row[10],
            source=row[11],
        )

    async def list_all(self, status: str | None = None) -> list[SkillEntry]:
        """
        List all registered skills, optionally filtered by status.

        Args:
            status: Filter by status (``active``, ``disabled``, ``failed``).
                    If None, returns all skills.
        """
        if status is not None:
            cursor = await self._db.execute(
                """
                SELECT id, name, description, file_path, status,
                       created_at, updated_at, usage_count, success_count,
                       failure_count, last_used_at, source
                FROM skills WHERE status = ?
                ORDER BY created_at DESC
                """,
                (status,),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT id, name, description, file_path, status,
                       created_at, updated_at, usage_count, success_count,
                       failure_count, last_used_at, source
                FROM skills
                ORDER BY created_at DESC
                """
            )

        rows = await cursor.fetchall()
        await cursor.close()

        return [
            SkillEntry(
                id=r[0],
                name=r[1],
                description=r[2],
                file_path=r[3],
                status=r[4],
                created_at=r[5],
                updated_at=r[6],
                usage_count=r[7],
                success_count=r[8],
                failure_count=r[9],
                last_used_at=r[10],
                source=r[11],
            )
            for r in rows
        ]

    async def update_stats(self, name: str, success: bool) -> None:
        """
        Update usage statistics after a skill execution.

        Increments ``usage_count`` and either ``success_count`` or
        ``failure_count``. Updates ``last_used_at`` and ``updated_at``.
        """
        now = time.time()
        if success:
            await self._db.execute(
                """
                UPDATE skills
                SET usage_count = usage_count + 1,
                    success_count = success_count + 1,
                    last_used_at = ?,
                    updated_at = ?
                WHERE name = ?
                """,
                (now, now, name),
            )
        else:
            await self._db.execute(
                """
                UPDATE skills
                SET usage_count = usage_count + 1,
                    failure_count = failure_count + 1,
                    last_used_at = ?,
                    updated_at = ?
                WHERE name = ?
                """,
                (now, now, name),
            )
        await self._db.commit()
        log.debug("skill_registry.stats_updated", name=name, success=success)

    async def delete(self, name: str) -> bool:
        """
        Delete a skill from the registry.

        Does NOT remove the file from disk (caller handles that).

        Returns:
            True if the skill was found and deleted.
        """
        cursor = await self._db.execute(
            "DELETE FROM skills WHERE name = ?", (name,)
        )
        await self._db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            log.info("skill_registry.deleted", name=name)
        else:
            log.warning("skill_registry.delete_not_found", name=name)
        return deleted

    async def set_status(self, name: str, status: str) -> None:
        """
        Update the status of a skill.

        Args:
            name:   Skill name.
            status: New status (``active``, ``disabled``, ``failed``).
        """
        now = time.time()
        await self._db.execute(
            "UPDATE skills SET status = ?, updated_at = ? WHERE name = ?",
            (status, now, name),
        )
        await self._db.commit()
        log.info("skill_registry.status_changed", name=name, status=status)
