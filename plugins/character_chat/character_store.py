"""
SQLite-backed persistence for character cards.

The store is a thin async wrapper around a single ``characters`` table. It's
decoupled from the plugin lifecycle so tests can inject a bare
``aiosqlite.Connection`` (see ``tests/test_character_store.py``).

All methods are coroutine-safe under a single connection: a per-store
:class:`asyncio.Lock` serialises writes so two ``attach_to_session`` calls
don't race each other on the JSON-blob ``active_sessions`` column.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterable

import aiosqlite
from pydantic import ValidationError

from .character_card import (
    AGE_STAGES,
    CharacterCard,
    CharacterCardError,
    parse_silly_tavern_card,
)


# SQL schema as a module constant so tests can reuse it.
SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    persona TEXT DEFAULT '',
    greeting TEXT DEFAULT '',
    scenario TEXT DEFAULT '',
    example_dialog TEXT DEFAULT '',
    avatar TEXT DEFAULT '',
    color TEXT DEFAULT '#7aa2f7',
    age_stage TEXT DEFAULT 'adult',
    voice TEXT DEFAULT '',
    relationships TEXT DEFAULT '{}',
    tags TEXT DEFAULT '[]',
    active_sessions TEXT DEFAULT '[]',
    proactive_pulse_pattern TEXT DEFAULT '',
    proactive_pulse_prompt TEXT DEFAULT '',
    archived INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_characters_name ON characters(name);
CREATE INDEX IF NOT EXISTS idx_characters_archived ON characters(archived);
"""

# Columns added after initial release — applied idempotently on schema init.
# Storing them here keeps upgrade SQL in one place (easier to reason about
# than scattered ALTER TABLE blocks spread across the store).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    # (column, ALTER statement)
    ("voice", "ALTER TABLE characters ADD COLUMN voice TEXT DEFAULT ''"),
)


# Columns that can be updated via ``update()``. Any column not in this set is
# silently ignored — that's the defensive default because the HTTP layer will
# sooner or later feed us unknown keys.
_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "name",
        "persona",
        "greeting",
        "scenario",
        "example_dialog",
        "avatar",
        "color",
        "age_stage",
        "voice",
        "relationships",
        "tags",
        "active_sessions",
        "proactive_pulse_pattern",
        "proactive_pulse_prompt",
        "archived",
    }
)


class CharacterStore:
    """Async CRUD over the ``characters`` table."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._db.row_factory = aiosqlite.Row
        self._lock = asyncio.Lock()

    # ─── Schema ──────────────────────────────────────────────────────────

    async def init_schema(self) -> None:
        """Create table + indexes if not present, and apply migrations."""
        await self._db.executescript(SCHEMA_SQL)
        # Apply any column-adds for existing databases. SQLite doesn't
        # have ``ADD COLUMN IF NOT EXISTS`` so we check PRAGMA first.
        cursor = await self._db.execute("PRAGMA table_info(characters)")
        existing_cols = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        for column, ddl in _MIGRATIONS:
            if column not in existing_cols:
                await self._db.execute(ddl)
        await self._db.commit()

    # ─── Basic CRUD ──────────────────────────────────────────────────────

    async def create(self, card: CharacterCard) -> CharacterCard:
        """Insert a new card.

        Raises :class:`CharacterCardError` on name collision (same name
        already exists and is not archived).
        """
        async with self._lock:
            existing = await self._fetch_by_name(card.name, include_archived=False)
            if existing is not None and existing.id != card.id:
                raise CharacterCardError(
                    f"character name {card.name!r} already in use by id "
                    f"{existing.id!r}"
                )
            row = card.to_row()
            cols = ",".join(row.keys())
            placeholders = ",".join("?" for _ in row)
            await self._db.execute(
                f"INSERT INTO characters ({cols}) VALUES ({placeholders})",
                tuple(row.values()),
            )
            await self._db.commit()
        return card

    async def get(self, character_id: str) -> CharacterCard | None:
        cursor = await self._db.execute(
            "SELECT * FROM characters WHERE id = ?", (character_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return CharacterCard.from_row(dict(row))

    async def get_by_name(
        self, name: str, *, include_archived: bool = False
    ) -> CharacterCard | None:
        return await self._fetch_by_name(name, include_archived=include_archived)

    async def list(
        self, *, include_archived: bool = False
    ) -> list[CharacterCard]:
        sql = "SELECT * FROM characters"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY created_at ASC"
        cursor = await self._db.execute(sql)
        rows = await cursor.fetchall()
        await cursor.close()
        return [CharacterCard.from_row(dict(r)) for r in rows]

    async def list_in_session(
        self, session_id: str, *, include_archived: bool = False
    ) -> list[CharacterCard]:
        """Return cards whose ``active_sessions`` contains ``session_id``.

        Implemented in Python rather than JSON-path SQL so it's portable
        across SQLite builds — the number of characters per system is tiny,
        so full-scan is fine.
        """
        all_cards = await self.list(include_archived=include_archived)
        return [c for c in all_cards if session_id in c.active_sessions]

    async def update(
        self, character_id: str, **patch: Any
    ) -> CharacterCard | None:
        """Update any subset of ``_UPDATABLE_COLUMNS`` on a card.

        Returns the updated card, or ``None`` if the id doesn't exist.
        Invalid age_stage raises :class:`CharacterCardError`; unknown keys
        are ignored.
        """
        if "age_stage" in patch and patch["age_stage"] not in AGE_STAGES:
            raise CharacterCardError(
                f"age_stage must be one of {AGE_STAGES!r}, got {patch['age_stage']!r}"
            )

        async with self._lock:
            current = await self.get(character_id)
            if current is None:
                return None

            merged = current.model_dump()
            for key, value in patch.items():
                if key not in _UPDATABLE_COLUMNS:
                    continue
                merged[key] = value
            merged["updated_at"] = time.time()

            try:
                new_card = CharacterCard(**merged)
            except CharacterCardError:
                raise
            except (ValidationError, ValueError) as exc:
                # Pydantic wraps field_validator errors in ValidationError; we
                # re-raise as CharacterCardError so callers have one exception
                # class to catch regardless of which validator fired.
                raise CharacterCardError(str(exc)) from exc

            # Name-collision guard if the name changed.
            if new_card.name != current.name:
                collision = await self._fetch_by_name(
                    new_card.name, include_archived=False
                )
                if collision is not None and collision.id != character_id:
                    raise CharacterCardError(
                        f"character name {new_card.name!r} already in use"
                    )

            row = new_card.to_row()
            cols = [c for c in row.keys() if c != "id"]
            assignments = ",".join(f"{c} = ?" for c in cols)
            values = [row[c] for c in cols]
            values.append(character_id)
            await self._db.execute(
                f"UPDATE characters SET {assignments} WHERE id = ?",
                tuple(values),
            )
            await self._db.commit()
        return new_card

    async def archive(self, character_id: str) -> bool:
        return bool(await self.update(character_id, archived=True))

    async def unarchive(self, character_id: str) -> bool:
        return bool(await self.update(character_id, archived=False))

    async def delete(self, character_id: str) -> bool:
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM characters WHERE id = ?", (character_id,)
            )
            deleted = cursor.rowcount or 0
            await cursor.close()
            await self._db.commit()
        return deleted > 0

    # ─── Session membership ──────────────────────────────────────────────

    async def attach_to_session(
        self, character_id: str, session_id: str
    ) -> CharacterCard | None:
        """Add ``session_id`` to the card's ``active_sessions`` list."""
        async with self._lock:
            card = await self.get(character_id)
            if card is None:
                return None
            if session_id in card.active_sessions:
                return card
            new_sessions = [*card.active_sessions, session_id]
        # Release lock for the actual update — update() re-acquires it.
        return await self.update(character_id, active_sessions=new_sessions)

    async def detach_from_session(
        self, character_id: str, session_id: str
    ) -> CharacterCard | None:
        async with self._lock:
            card = await self.get(character_id)
            if card is None:
                return None
            if session_id not in card.active_sessions:
                return card
            new_sessions = [s for s in card.active_sessions if s != session_id]
        return await self.update(character_id, active_sessions=new_sessions)

    # ─── Silly-Tavern import ─────────────────────────────────────────────

    async def import_silly_tavern(
        self,
        payload: dict[str, Any],
        *,
        color: str | None = None,
        age_stage: str = "adult",
    ) -> CharacterCard:
        """Parse a Silly-Tavern card payload and persist it."""
        card = parse_silly_tavern_card(payload)
        if color:
            card.color = color
        card.age_stage = age_stage
        return await self.create(card)

    # ─── Helpers ─────────────────────────────────────────────────────────

    async def _fetch_by_name(
        self, name: str, *, include_archived: bool
    ) -> CharacterCard | None:
        sql = "SELECT * FROM characters WHERE name = ?"
        params: tuple[Any, ...] = (name,)
        if not include_archived:
            sql += " AND archived = 0"
        sql += " LIMIT 1"
        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return CharacterCard.from_row(dict(row))

    async def count(self, *, include_archived: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM characters"
        if not include_archived:
            sql += " WHERE archived = 0"
        cursor = await self._db.execute(sql)
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0

    async def bulk_insert(self, cards: Iterable[CharacterCard]) -> int:
        """Insert multiple cards (test/fixture helper). Skips collisions."""
        inserted = 0
        for card in cards:
            try:
                await self.create(card)
                inserted += 1
            except CharacterCardError:
                continue
        return inserted
