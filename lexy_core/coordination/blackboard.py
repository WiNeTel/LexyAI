"""
Lexy AI - Coordination: Blackboard.

The shared "schwarzes Brett" for multi-agent coordination. Two kinds of
shared state live here:

* **Posts** — an append-only feed of what agents discover and decide:
  findings, successes, dead-ends, demands, comments, decisions. The
  ``dead_end`` kind is deliberately part of the vocabulary: it is the
  cross-team "do not walk into this wall again" register from the
  AutoScientists paper, readable by everyone in the same scope.
* **Facts** — a per-scope key-value store for the current shared truth
  (e.g. world-state snapshots, the active decision).

A ``scope`` string isolates independent arenas so one process can host
many boards at once: one RP scene, one expert-panel session, a future
MCP-Town island. Persisted via aiosqlite (native async — no
``run_in_executor``), so the board survives restarts.

The :class:`Blackboard` takes an already-open ``aiosqlite.Connection``
(dependency injection, same pattern as ``orchestrator.TaskQueue`` and
``scheduler``); the owning plugin/app opens and closes it.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.blackboard")

# Canonical post kinds. Unknown kinds are still stored (forgiving), but
# logged — so a typo surfaces without silently dropping a post.
POST_KINDS: frozenset[str] = frozenset(
    {"finding", "success", "dead_end", "demand", "comment", "decision"}
)


@dataclass
class Post:
    """A single entry on the blackboard."""

    id: str
    scope: str
    author: str
    kind: str
    body: str
    meta: dict[str, Any]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "author": self.author,
            "kind": self.kind,
            "body": self.body,
            "meta": self.meta,
            "created_at": self.created_at,
        }


class Blackboard:
    """SQLite-backed shared board: append-only posts + per-scope facts."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def init_tables(self) -> None:
        """Create the ``posts`` and ``facts`` tables if they do not exist."""
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS posts (
                id          TEXT PRIMARY KEY,
                scope       TEXT NOT NULL,
                author      TEXT NOT NULL,
                kind        TEXT NOT NULL,
                body        TEXT NOT NULL,
                meta_json   TEXT NOT NULL DEFAULT '{}',
                created_at  REAL NOT NULL
            )"""
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_posts_scope "
            "ON posts(scope, created_at ASC)"
        )
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS facts (
                scope       TEXT NOT NULL,
                key         TEXT NOT NULL,
                value_json  TEXT NOT NULL,
                updated_at  REAL NOT NULL,
                PRIMARY KEY (scope, key)
            )"""
        )
        await self._db.commit()
        log.info("blackboard.tables_ready")

    # ─── Posts (append-only feed) ────────────────────────────────────

    async def post(
        self,
        scope: str,
        author: str,
        kind: str,
        body: str,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Append a post to ``scope`` and return its generated id.

        ``kind`` should be one of :data:`POST_KINDS`; unknown kinds are
        still stored but logged so typos surface.
        """
        if kind not in POST_KINDS:
            log.warning("blackboard.unknown_kind", kind=kind, scope=scope)

        post_id = uuid.uuid4().hex[:12]
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        await self._db.execute(
            "INSERT INTO posts (id, scope, author, kind, body, meta_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (post_id, scope, author, kind, body, meta_json, time.time()),
        )
        await self._db.commit()
        log.info(
            "blackboard.post", scope=scope, author=author, kind=kind, post_id=post_id
        )
        return post_id

    async def read(
        self,
        scope: str,
        since: float = 0.0,
        kinds: Iterable[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Read posts in ``scope`` created after ``since`` (oldest first).

        Optionally filter by ``kinds`` (e.g. ``["dead_end"]`` to scan the
        dead-end register before starting a new direction).
        """
        sql = "SELECT id, scope, author, kind, body, meta_json, created_at FROM posts WHERE scope = ? AND created_at > ?"
        params: list[Any] = [scope, since]
        kind_list = [k for k in (kinds or [])]
        if kind_list:
            placeholders = ",".join("?" for _ in kind_list)
            sql += f" AND kind IN ({placeholders})"
            params.extend(kind_list)
        sql += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)

        async with self._db.execute(sql, params) as cur:
            rows = await cur.fetchall()

        return [self._row_to_post(row).to_dict() for row in rows]

    # ─── Facts (per-scope key-value) ─────────────────────────────────

    async def set_fact(self, scope: str, key: str, value: Any) -> None:
        """Set (upsert) a shared fact. ``value`` must be JSON-serialisable."""
        value_json = json.dumps(value, ensure_ascii=False)
        await self._db.execute(
            "INSERT INTO facts (scope, key, value_json, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(scope, key) DO UPDATE SET "
            "value_json = excluded.value_json, updated_at = excluded.updated_at",
            (scope, key, value_json, time.time()),
        )
        await self._db.commit()
        log.debug("blackboard.set_fact", scope=scope, key=key)

    async def get_fact(self, scope: str, key: str, default: Any = None) -> Any:
        """Return a shared fact, or ``default`` if unset."""
        async with self._db.execute(
            "SELECT value_json FROM facts WHERE scope = ? AND key = ?",
            (scope, key),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            log.warning("blackboard.fact_decode_failed", scope=scope, key=key)
            return default

    async def snapshot(self, scope: str) -> dict[str, Any]:
        """Return all current facts in ``scope`` as a plain dict."""
        async with self._db.execute(
            "SELECT key, value_json FROM facts WHERE scope = ?",
            (scope,),
        ) as cur:
            rows = await cur.fetchall()
        out: dict[str, Any] = {}
        for key, value_json in rows:
            try:
                out[key] = json.loads(value_json)
            except (json.JSONDecodeError, TypeError):
                out[key] = None
        return out

    # ─── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _row_to_post(row: Any) -> Post:
        try:
            meta = json.loads(row[5]) if row[5] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        return Post(
            id=row[0],
            scope=row[1],
            author=row[2],
            kind=row[3],
            body=row[4],
            meta=meta,
            created_at=row[6],
        )
