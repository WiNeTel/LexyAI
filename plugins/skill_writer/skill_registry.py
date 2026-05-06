"""
Lexy AI — Skill Registry (Phase 11 — agentskills.io aware).

SQLite-backed registry of every installed skill plus its metadata, lifecycle
status, and usage stats. Phase 11 swapped the on-disk format from a flat
``data/skills/<name>.py`` to a per-skill folder (`data/skills/<name>/`)
matching the agentskills.io spec — this module follows along:

* ``file_path`` now holds the **folder** path, not a single ``.py`` file.
* New columns surface the spec's frontmatter fields (``license``,
  ``compatibility``, ``metadata_json``, ``allowed_tools``, ``body_md``).
* ``scan_disk()`` walks subdirectories instead of ``*.py`` siblings.

Schema migrations are idempotent — ``ALTER TABLE`` calls swallow the
"column already exists" error so existing DBs upgrade in place. We
support legacy entries (where ``file_path`` still points to a single
file) by leaving them alone; the loader will surface them as broken
rows so the user can re-import.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

from .skill_loader import SkillLoaderError, discover_skills, load_skill_folder
from .skill_spec import SkillSpecError

log = get_logger(module="skill_registry")


@dataclass
class SkillEntry:
    """One registered skill with its metadata, lifecycle, and stats.

    Phase 11 added five frontmatter-mirroring fields. Older entries
    written before the migration get sensible defaults
    (empty/None/empty-dict) so existing callers don't crash.
    """

    id: str
    name: str
    description: str
    file_path: str  # path to the skill folder (not the .py file)
    status: str  # active, disabled, failed
    created_at: float
    updated_at: float | None
    usage_count: int
    success_count: int
    failure_count: int
    last_used_at: float | None
    source: str  # auto, manual, sub_agent, imported

    # ── Phase 11 (agentskills.io frontmatter) ────────────────────
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None
    body_md: str = ""

    def to_public(self) -> dict[str, Any]:
        """JSON-friendly dict for REST/WS responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_used_at": self.last_used_at,
            "source": self.source,
            "license": self.license,
            "compatibility": self.compatibility,
            "metadata": dict(self.metadata),
            "allowed_tools": self.allowed_tools,
        }


# Column list used by every SELECT — kept in one place so adding a
# field in the future is a single-line change.
_SELECT_COLS = (
    "id, name, description, file_path, status, "
    "created_at, updated_at, usage_count, success_count, "
    "failure_count, last_used_at, source, "
    "license, compatibility, metadata_json, allowed_tools, body_md"
)


def _row_to_entry(row: tuple[Any, ...]) -> SkillEntry:
    """Convert one ``SELECT`` row into a :class:`SkillEntry`."""
    metadata: dict[str, str] = {}
    raw_meta = row[14]
    if raw_meta:
        try:
            parsed = json.loads(raw_meta)
            if isinstance(parsed, dict):
                metadata = {str(k): str(v) for k, v in parsed.items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}

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
        license=row[12],
        compatibility=row[13],
        metadata=metadata,
        allowed_tools=row[15],
        body_md=row[16] or "",
    )


# Frontmatter-mirroring columns added in Phase 11. Listed in the
# order they should appear at the end of the table.
_PHASE_11_COLUMNS: tuple[tuple[str, str], ...] = (
    ("license", "TEXT"),
    ("compatibility", "TEXT"),
    ("metadata_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("allowed_tools", "TEXT"),
    ("body_md", "TEXT NOT NULL DEFAULT ''"),
)


class SkillRegistry:
    """Manages the lifecycle of skill folders.

    Backed by an :mod:`aiosqlite` connection (shared with the rest of
    the plugin's DB). Phase 11 adds folder-aware ``scan_disk`` and
    columns for the agentskills.io frontmatter fields.
    """

    def __init__(
        self, db: aiosqlite.Connection, skills_path: Path
    ) -> None:
        self._db = db
        self._skills_path = skills_path

    async def init_tables(self) -> None:
        """Create the schema (or migrate it forward).

        Safe to call multiple times. New columns are added via ``ALTER
        TABLE`` if they're missing.
        """
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
                source      TEXT NOT NULL DEFAULT 'manual',
                license     TEXT,
                compatibility TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                allowed_tools TEXT,
                body_md     TEXT NOT NULL DEFAULT ''
            )
            """
        )
        # Idempotent migration for DBs that pre-date Phase 11 — try
        # to ALTER, swallow the "duplicate column" error if it's
        # already there.
        for col_name, col_decl in _PHASE_11_COLUMNS:
            try:
                await self._db.execute(
                    f"ALTER TABLE skills ADD COLUMN {col_name} {col_decl}"
                )
            except aiosqlite.OperationalError as exc:
                msg = str(exc).lower()
                if "duplicate column" not in msg and "already exists" not in msg:
                    raise

        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status)"
        )
        await self._db.commit()
        log.info("skill_registry.tables_ready")

    async def scan_disk(self) -> int:
        """Discover skill folders on disk and register any new ones.

        Phase 11 walks ``skills_path`` for *folders* (each containing
        a SKILL.md). Folders that fail validation are logged but don't
        abort the scan — the per-folder load is wrapped in a defensive
        try-block.

        Returns the number of newly registered skills.
        """
        if not self._skills_path.exists():
            return 0

        cards = await discover_skills(self._skills_path)
        added = 0
        for card in cards:
            existing = await self.get(card.name)
            if existing is not None:
                continue
            try:
                await self.register(
                    name=card.name,
                    description=card.description,
                    file_path=str(card.folder),
                    source="manual",
                    license=card.frontmatter.license,
                    compatibility=card.frontmatter.compatibility,
                    metadata=card.frontmatter.metadata,
                    allowed_tools=card.frontmatter.allowed_tools,
                    body_md=card.frontmatter.body,
                )
                added += 1
                log.info(
                    "skill_registry.scanned",
                    name=card.name,
                    folder=str(card.folder),
                )
            except (SkillLoaderError, SkillSpecError, OSError) as exc:
                log.warning(
                    "skill_registry.scan_register_failed",
                    name=card.name,
                    error=str(exc),
                )
        return added

    async def register(
        self,
        name: str,
        description: str,
        file_path: str,
        source: str = "manual",
        *,
        license: str | None = None,
        compatibility: str | None = None,
        metadata: dict[str, str] | None = None,
        allowed_tools: str | None = None,
        body_md: str = "",
    ) -> str:
        """Register a new skill folder.

        Args:
            name:        Spec-valid skill name (= folder name).
            description: 1-1024 chars description from the frontmatter.
            file_path:   Absolute path to the **skill folder**.
            source:      Origin: ``auto``, ``manual``, ``sub_agent``,
                         or ``imported`` (Phase 11.B).
            license, compatibility, metadata, allowed_tools, body_md:
                         Mirror of the agentskills.io frontmatter
                         fields. Persisted so the UI doesn't need to
                         re-read SKILL.md to render the entry.
        """
        skill_id = uuid.uuid4().hex[:12]
        now = time.time()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        await self._db.execute(
            """
            INSERT INTO skills
                (id, name, description, file_path, status, created_at, source,
                 license, compatibility, metadata_json, allowed_tools, body_md)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id, name, description, file_path, now, source,
                license, compatibility, metadata_json, allowed_tools, body_md,
            ),
        )
        await self._db.commit()
        log.info(
            "skill_registry.registered",
            id=skill_id,
            name=name,
            source=source,
        )
        return skill_id

    async def update_metadata(
        self,
        name: str,
        *,
        description: str | None = None,
        license: str | None = None,
        compatibility: str | None = None,
        metadata: dict[str, str] | None = None,
        allowed_tools: str | None = None,
        body_md: str | None = None,
    ) -> bool:
        """Patch frontmatter-derived fields on an existing entry.

        Used by the importer when re-importing a skill with
        ``overwrite=True`` so the registry doesn't drift from disk.
        """
        sets: list[str] = []
        params: list[Any] = []
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if license is not None:
            sets.append("license = ?")
            params.append(license)
        if compatibility is not None:
            sets.append("compatibility = ?")
            params.append(compatibility)
        if metadata is not None:
            sets.append("metadata_json = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))
        if allowed_tools is not None:
            sets.append("allowed_tools = ?")
            params.append(allowed_tools)
        if body_md is not None:
            sets.append("body_md = ?")
            params.append(body_md)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(time.time())
        params.append(name)
        cur = await self._db.execute(
            f"UPDATE skills SET {', '.join(sets)} WHERE name = ?",
            tuple(params),
        )
        await self._db.commit()
        return (cur.rowcount or 0) > 0

    async def get(self, name: str) -> SkillEntry | None:
        """Look up a skill by name. Returns ``None`` if not found."""
        cursor = await self._db.execute(
            f"SELECT {_SELECT_COLS} FROM skills WHERE name = ?",
            (name,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return _row_to_entry(row) if row else None

    async def list_all(self, status: str | None = None) -> list[SkillEntry]:
        """List all registered skills, optionally filtered by status."""
        if status is not None:
            cursor = await self._db.execute(
                f"SELECT {_SELECT_COLS} FROM skills WHERE status = ? "
                "ORDER BY created_at DESC",
                (status,),
            )
        else:
            cursor = await self._db.execute(
                f"SELECT {_SELECT_COLS} FROM skills "
                "ORDER BY created_at DESC"
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_entry(r) for r in rows]

    async def update_stats(self, name: str, success: bool) -> None:
        """Update usage statistics after a skill execution."""
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
        """Delete a skill from the registry. (Disk cleanup is the caller's job.)"""
        cursor = await self._db.execute(
            "DELETE FROM skills WHERE name = ?", (name,)
        )
        await self._db.commit()
        deleted = (cursor.rowcount or 0) > 0
        if deleted:
            log.info("skill_registry.deleted", name=name)
        else:
            log.warning("skill_registry.delete_not_found", name=name)
        return deleted

    async def set_status(self, name: str, status: str) -> None:
        """Update the lifecycle status of a skill."""
        now = time.time()
        await self._db.execute(
            "UPDATE skills SET status = ?, updated_at = ? WHERE name = ?",
            (status, now, name),
        )
        await self._db.commit()
        log.info("skill_registry.status_changed", name=name, status=status)

    async def load_card(self, name: str):
        """Convenience: registry entry → live :class:`SkillCard` from disk.

        Returns ``None`` if the entry is missing or the folder no
        longer exists. Used by ``run_skill`` so it doesn't have to
        re-walk the skills root.
        """
        entry = await self.get(name)
        if entry is None:
            return None
        folder = Path(entry.file_path)
        if not folder.is_dir():
            log.warning(
                "skill_registry.folder_missing",
                name=name,
                path=str(folder),
            )
            return None
        try:
            return await load_skill_folder(folder)
        except (SkillLoaderError, SkillSpecError) as exc:
            log.warning(
                "skill_registry.load_card_failed",
                name=name,
                error=str(exc),
            )
            return None
