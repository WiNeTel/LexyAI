"""
Lorebook persistence for the character_chat plugin.

A *lorebook* is a named collection of *entries* — each entry has trigger
keys, content text, and a small bag of placement / priority knobs. When
a character round runs, the engine scans recent chat (and optionally the
character state) for entries whose keys match, sorts them by priority,
and injects the matching content as a prompt section.

The model deliberately mirrors SillyTavern's "World Info" mechanic so
imports between the two systems stay easy:

* **Scope**: a lorebook belongs to either a single character, a single
  session, or is global (visible to every round). Multiple lorebooks
  can be active at once — the engine merges entries.
* **Triggers**: each entry has 0–N case-insensitive substring keys.
  Empty keys + ``always_on=True`` means the entry fires every round.
* **Position**: where the matched content lands in the prompt. We map
  the SillyTavern slots to Lexy's section layout.
* **Priority**: lower number = inserted first (closer to the top of
  the lore block); ties broken alphabetically by entry name.
* **Token budget**: total characters of lore content per round, capped
  in :class:`LorebookEngine` (not here).

Persistence uses two SQLite tables in the existing character_chat DB:

```
lorebooks(id, name, description, scope, scope_id, enabled,
          token_budget, created_at, updated_at)
lore_entries(id, lorebook_id, name, keys_json, content,
             position, priority, always_on, scan_depth,
             enabled, created_at, updated_at)
```
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import aiosqlite


# ─── Constants ──────────────────────────────────────────────────────


SCOPE_GLOBAL = "global"
SCOPE_CHARACTER = "character"
SCOPE_SESSION = "session"
VALID_SCOPES: tuple[str, ...] = (SCOPE_GLOBAL, SCOPE_CHARACTER, SCOPE_SESSION)

# Where in the prompt the matched lore content lands. The values map
# onto sections the orchestrator's prompt builder already understands;
# new positions go through `_build_turn_sections` to find their slot.
POSITION_BEFORE_PERSONA = "before_persona"
POSITION_AFTER_PERSONA = "after_persona"
POSITION_BEFORE_SCENARIO = "before_scenario"   # default — feels like world-text
POSITION_BEFORE_HISTORY = "before_history"
POSITION_BEFORE_USER_MESSAGE = "before_user_message"
VALID_POSITIONS: tuple[str, ...] = (
    POSITION_BEFORE_PERSONA,
    POSITION_AFTER_PERSONA,
    POSITION_BEFORE_SCENARIO,
    POSITION_BEFORE_HISTORY,
    POSITION_BEFORE_USER_MESSAGE,
)

# Default token-budget for lore content per round. The engine truncates
# the total content above this; entries are kept whole until the cap.
DEFAULT_TOKEN_BUDGET = 1500


# ─── Models ─────────────────────────────────────────────────────────


@dataclass
class Lorebook:
    """One collection of lore entries with a scope + budget."""

    id: str
    name: str
    description: str = ""
    scope: str = SCOPE_GLOBAL
    scope_id: str = ""           # character_id / session_id; "" for global
    enabled: bool = True
    token_budget: int = DEFAULT_TOKEN_BUDGET
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "enabled": 1 if self.enabled else 0,
            "token_budget": int(self.token_budget),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Lorebook":
        return cls(
            id=str(row["id"]),
            name=str(row.get("name") or ""),
            description=str(row.get("description") or ""),
            scope=str(row.get("scope") or SCOPE_GLOBAL),
            scope_id=str(row.get("scope_id") or ""),
            enabled=bool(row.get("enabled")),
            token_budget=int(row.get("token_budget") or DEFAULT_TOKEN_BUDGET),
            created_at=float(row.get("created_at") or 0.0),
            updated_at=float(row.get("updated_at") or 0.0),
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "scope_id": self.scope_id,
            "enabled": self.enabled,
            "token_budget": self.token_budget,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class LoreEntry:
    """One entry within a lorebook."""

    id: str
    lorebook_id: str
    name: str
    keys: list[str] = field(default_factory=list)
    content: str = ""
    position: str = POSITION_BEFORE_SCENARIO
    priority: int = 100              # lower → earlier in the lore block
    always_on: bool = False           # fires every round, no key match needed
    scan_depth: int = 4               # last N messages scanned for keys
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lorebook_id": self.lorebook_id,
            "name": self.name,
            "keys_json": json.dumps(list(self.keys), ensure_ascii=False),
            "content": self.content,
            "position": self.position,
            "priority": int(self.priority),
            "always_on": 1 if self.always_on else 0,
            "scan_depth": int(self.scan_depth),
            "enabled": 1 if self.enabled else 0,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "LoreEntry":
        keys: list[str] = []
        keys_raw = row.get("keys_json") or "[]"
        try:
            parsed = json.loads(keys_raw)
            if isinstance(parsed, list):
                keys = [str(k) for k in parsed if str(k).strip()]
        except (TypeError, ValueError, json.JSONDecodeError):
            keys = []
        return cls(
            id=str(row["id"]),
            lorebook_id=str(row["lorebook_id"]),
            name=str(row.get("name") or ""),
            keys=keys,
            content=str(row.get("content") or ""),
            position=str(row.get("position") or POSITION_BEFORE_SCENARIO),
            priority=int(row.get("priority") or 100),
            always_on=bool(row.get("always_on")),
            scan_depth=int(row.get("scan_depth") or 4),
            enabled=bool(row.get("enabled")),
            created_at=float(row.get("created_at") or 0.0),
            updated_at=float(row.get("updated_at") or 0.0),
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lorebook_id": self.lorebook_id,
            "name": self.name,
            "keys": list(self.keys),
            "content": self.content,
            "position": self.position,
            "priority": self.priority,
            "always_on": self.always_on,
            "scan_depth": self.scan_depth,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ─── Store ──────────────────────────────────────────────────────────


class LorebookStore:
    """Async CRUD over the ``lorebooks`` + ``lore_entries`` tables."""

    SCHEMA_LOREBOOKS = """
    CREATE TABLE IF NOT EXISTS lorebooks (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        description   TEXT NOT NULL DEFAULT '',
        scope         TEXT NOT NULL DEFAULT 'global',
        scope_id      TEXT NOT NULL DEFAULT '',
        enabled       INTEGER NOT NULL DEFAULT 1,
        token_budget  INTEGER NOT NULL DEFAULT 1500,
        created_at    REAL NOT NULL,
        updated_at    REAL NOT NULL
    )
    """
    SCHEMA_ENTRIES = """
    CREATE TABLE IF NOT EXISTS lore_entries (
        id           TEXT PRIMARY KEY,
        lorebook_id  TEXT NOT NULL,
        name         TEXT NOT NULL,
        keys_json    TEXT NOT NULL DEFAULT '[]',
        content      TEXT NOT NULL DEFAULT '',
        position     TEXT NOT NULL DEFAULT 'before_scenario',
        priority     INTEGER NOT NULL DEFAULT 100,
        always_on    INTEGER NOT NULL DEFAULT 0,
        scan_depth   INTEGER NOT NULL DEFAULT 4,
        enabled      INTEGER NOT NULL DEFAULT 1,
        created_at   REAL NOT NULL,
        updated_at   REAL NOT NULL,
        FOREIGN KEY(lorebook_id) REFERENCES lorebooks(id) ON DELETE CASCADE
    )
    """
    INDEX_BOOKS = (
        "CREATE INDEX IF NOT EXISTS idx_lorebooks_scope ON lorebooks(scope, scope_id)",
        "CREATE INDEX IF NOT EXISTS idx_lorebooks_enabled ON lorebooks(enabled)",
    )
    INDEX_ENTRIES = (
        "CREATE INDEX IF NOT EXISTS idx_lore_entries_book ON lore_entries(lorebook_id)",
        "CREATE INDEX IF NOT EXISTS idx_lore_entries_enabled ON lore_entries(enabled)",
    )

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db
        self._db.row_factory = aiosqlite.Row
        self._lock = asyncio.Lock()

    async def init_schema(self) -> None:
        await self._db.execute(self.SCHEMA_LOREBOOKS)
        await self._db.execute(self.SCHEMA_ENTRIES)
        for stmt in (*self.INDEX_BOOKS, *self.INDEX_ENTRIES):
            await self._db.execute(stmt)
        await self._db.commit()

    # ─── Lorebook CRUD ───────────────────────────────────────────────

    async def create_lorebook(
        self,
        *,
        name: str,
        description: str = "",
        scope: str = SCOPE_GLOBAL,
        scope_id: str = "",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> Lorebook:
        if scope not in VALID_SCOPES:
            raise ValueError(f"unknown scope: {scope!r}")
        if scope == SCOPE_GLOBAL and scope_id:
            scope_id = ""  # global lorebooks ignore scope_id
        if scope != SCOPE_GLOBAL and not scope_id:
            raise ValueError(
                f"scope={scope!r} requires a scope_id (character_id or session_id)"
            )
        book = Lorebook(
            id=uuid.uuid4().hex[:12],
            name=name.strip() or "untitled",
            description=description.strip(),
            scope=scope,
            scope_id=scope_id,
            token_budget=max(0, int(token_budget)),
        )
        async with self._lock:
            row = book.to_row()
            await self._db.execute(
                "INSERT INTO lorebooks "
                "(id, name, description, scope, scope_id, enabled, "
                "token_budget, created_at, updated_at) "
                "VALUES (:id, :name, :description, :scope, :scope_id, "
                ":enabled, :token_budget, :created_at, :updated_at)",
                row,
            )
            await self._db.commit()
        return book

    async def get_lorebook(self, lorebook_id: str) -> Lorebook | None:
        async with self._db.execute(
            "SELECT * FROM lorebooks WHERE id = ?", (lorebook_id,)
        ) as cur:
            row = await cur.fetchone()
        return Lorebook.from_row(dict(row)) if row else None

    async def list_lorebooks(
        self,
        *,
        scope: str | None = None,
        scope_id: str | None = None,
        enabled_only: bool = False,
    ) -> list[Lorebook]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if enabled_only:
            clauses.append("enabled = 1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM lorebooks {where} ORDER BY name COLLATE NOCASE ASC"
        async with self._db.execute(sql, params) as cur:
            rows = list(await cur.fetchall())
        return [Lorebook.from_row(dict(r)) for r in rows]

    async def update_lorebook(
        self,
        lorebook_id: str,
        **patch: Any,
    ) -> Lorebook | None:
        existing = await self.get_lorebook(lorebook_id)
        if existing is None:
            return None
        allowed = {
            "name", "description", "scope", "scope_id",
            "enabled", "token_budget",
        }
        cleaned: dict[str, Any] = {}
        for key, value in patch.items():
            if key not in allowed:
                continue
            if key == "scope" and value not in VALID_SCOPES:
                raise ValueError(f"unknown scope: {value!r}")
            cleaned[key] = value
        if not cleaned:
            return existing
        merged = existing.to_row()
        for k, v in cleaned.items():
            if k == "enabled":
                merged[k] = 1 if v else 0
            elif k == "token_budget":
                merged[k] = max(0, int(v))
            else:
                merged[k] = v
        merged["updated_at"] = time.time()
        sets = ", ".join(f"{k} = :{k}" for k in cleaned.keys()) + ", updated_at = :updated_at"
        async with self._lock:
            await self._db.execute(
                f"UPDATE lorebooks SET {sets} WHERE id = :id",
                {**cleaned, "id": lorebook_id, "updated_at": merged["updated_at"]},
            )
            await self._db.commit()
        return await self.get_lorebook(lorebook_id)

    async def delete_lorebook(self, lorebook_id: str) -> bool:
        async with self._lock:
            cur = await self._db.execute(
                "DELETE FROM lorebooks WHERE id = ?", (lorebook_id,)
            )
            # Cascade: ON DELETE CASCADE only fires when foreign_keys=ON,
            # which isn't the default for SQLite. We delete entries
            # explicitly to keep the contract simple.
            await self._db.execute(
                "DELETE FROM lore_entries WHERE lorebook_id = ?",
                (lorebook_id,),
            )
            await self._db.commit()
        return (cur.rowcount or 0) > 0

    # ─── LoreEntry CRUD ──────────────────────────────────────────────

    async def create_entry(
        self,
        *,
        lorebook_id: str,
        name: str,
        keys: Iterable[str] = (),
        content: str = "",
        position: str = POSITION_BEFORE_SCENARIO,
        priority: int = 100,
        always_on: bool = False,
        scan_depth: int = 4,
    ) -> LoreEntry:
        if position not in VALID_POSITIONS:
            raise ValueError(f"unknown position: {position!r}")
        clean_keys = [
            str(k).strip() for k in (keys or []) if str(k).strip()
        ]
        if not clean_keys and not always_on:
            raise ValueError(
                "entry must have at least one key OR always_on=True"
            )
        # Defensive: scope check — the lorebook must exist.
        book = await self.get_lorebook(lorebook_id)
        if book is None:
            raise ValueError(f"lorebook not found: {lorebook_id!r}")
        entry = LoreEntry(
            id=uuid.uuid4().hex[:12],
            lorebook_id=lorebook_id,
            name=name.strip() or "entry",
            keys=clean_keys,
            content=content,
            position=position,
            priority=int(priority),
            always_on=bool(always_on),
            scan_depth=max(0, int(scan_depth)),
        )
        async with self._lock:
            await self._db.execute(
                "INSERT INTO lore_entries "
                "(id, lorebook_id, name, keys_json, content, position, "
                "priority, always_on, scan_depth, enabled, "
                "created_at, updated_at) "
                "VALUES (:id, :lorebook_id, :name, :keys_json, :content, "
                ":position, :priority, :always_on, :scan_depth, :enabled, "
                ":created_at, :updated_at)",
                entry.to_row(),
            )
            await self._db.commit()
        return entry

    async def get_entry(self, entry_id: str) -> LoreEntry | None:
        async with self._db.execute(
            "SELECT * FROM lore_entries WHERE id = ?", (entry_id,),
        ) as cur:
            row = await cur.fetchone()
        return LoreEntry.from_row(dict(row)) if row else None

    async def list_entries(
        self,
        *,
        lorebook_id: str | None = None,
        enabled_only: bool = False,
    ) -> list[LoreEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if lorebook_id is not None:
            clauses.append("lorebook_id = ?")
            params.append(lorebook_id)
        if enabled_only:
            clauses.append("enabled = 1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            f"SELECT * FROM lore_entries {where} "
            f"ORDER BY priority ASC, name COLLATE NOCASE ASC"
        )
        async with self._db.execute(sql, params) as cur:
            rows = list(await cur.fetchall())
        return [LoreEntry.from_row(dict(r)) for r in rows]

    async def update_entry(
        self,
        entry_id: str,
        **patch: Any,
    ) -> LoreEntry | None:
        existing = await self.get_entry(entry_id)
        if existing is None:
            return None
        allowed = {
            "name", "content", "position", "priority", "always_on",
            "scan_depth", "enabled", "keys",
        }
        cleaned: dict[str, Any] = {}
        for key, value in patch.items():
            if key not in allowed:
                continue
            if key == "position" and value not in VALID_POSITIONS:
                raise ValueError(f"unknown position: {value!r}")
            cleaned[key] = value
        if not cleaned:
            return existing
        # Materialise the row update.
        row = existing.to_row()
        for k, v in cleaned.items():
            if k == "keys":
                row["keys_json"] = json.dumps(
                    [str(x).strip() for x in (v or []) if str(x).strip()],
                    ensure_ascii=False,
                )
            elif k in ("always_on", "enabled"):
                row[k] = 1 if v else 0
            elif k in ("priority", "scan_depth"):
                row[k] = max(0, int(v))
            else:
                row[k] = v
        row["updated_at"] = time.time()
        cols = [
            c for c in row.keys()
            if c not in ("id", "lorebook_id", "created_at")
        ]
        sets = ", ".join(f"{c} = :{c}" for c in cols)
        async with self._lock:
            await self._db.execute(
                f"UPDATE lore_entries SET {sets} WHERE id = :id",
                {**{c: row[c] for c in cols}, "id": entry_id},
            )
            await self._db.commit()
        return await self.get_entry(entry_id)

    async def delete_entry(self, entry_id: str) -> bool:
        async with self._lock:
            cur = await self._db.execute(
                "DELETE FROM lore_entries WHERE id = ?", (entry_id,)
            )
            await self._db.commit()
        return (cur.rowcount or 0) > 0
