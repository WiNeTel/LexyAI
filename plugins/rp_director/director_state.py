"""
Director session state — SQLite persistence for in-flight RP setups.

A "Director session" is the period between :func:`start_rp_setup` (user
opened a setup dialog) and either :func:`commit_rp_setup` (Director writes
characters + activates character_mode) or :func:`cancel_rp_setup` (user
backed out). The state is small but must survive a backend restart so
mid-setup chats can resume after a crash.

States
------
- ``collecting`` — Director is gathering ideas; nothing committed yet.
- ``proposing``  — Director has at least one scenario or character draft
  in ``scenario_json`` / ``characters_json`` ready to commit.
- ``committed``  — final state after a successful ``commit_rp_setup``.
  The row is kept (not deleted) so we have an audit trail.
- ``cancelled``  — user aborted; row kept for the same reason.
"""

from __future__ import annotations

import json
import time
from typing import Any

import aiosqlite

VALID_STATES = ("collecting", "proposing", "committed", "cancelled")


class DirectorState:
    """Thin async wrapper around the ``rp_director_sessions`` table."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def init_table(self) -> None:
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS rp_director_sessions (
                session_id      TEXT PRIMARY KEY,
                state           TEXT NOT NULL,
                scenario_json   TEXT,
                characters_json TEXT,
                user_intent     TEXT,
                started_at      REAL NOT NULL,
                updated_at      REAL NOT NULL
            )"""
        )
        await self._db.commit()

    async def start(self, session_id: str, user_intent: str = "") -> dict[str, Any]:
        now = time.time()
        await self._db.execute(
            """INSERT INTO rp_director_sessions
                   (session_id, state, scenario_json, characters_json,
                    user_intent, started_at, updated_at)
               VALUES (?, 'collecting', NULL, NULL, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   state='collecting',
                   scenario_json=NULL,
                   characters_json=NULL,
                   user_intent=excluded.user_intent,
                   started_at=excluded.started_at,
                   updated_at=excluded.updated_at""",
            (session_id, user_intent, now, now),
        )
        await self._db.commit()
        return {
            "session_id": session_id,
            "state": "collecting",
            "user_intent": user_intent,
            "started_at": now,
            "updated_at": now,
            "scenario": None,
            "characters": [],
        }

    async def get(self, session_id: str) -> dict[str, Any] | None:
        async with self._db.execute(
            "SELECT session_id, state, scenario_json, characters_json, "
            "user_intent, started_at, updated_at "
            "FROM rp_director_sessions WHERE session_id = ?",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    async def is_active(self, session_id: str) -> bool:
        """True iff Director currently owns the prompt for ``session_id``."""
        info = await self.get(session_id)
        return bool(info and info["state"] in ("collecting", "proposing"))

    async def set_scenario(
        self, session_id: str, scenario: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not isinstance(scenario, dict):
            return None
        now = time.time()
        await self._db.execute(
            "UPDATE rp_director_sessions "
            "SET scenario_json = ?, state = 'proposing', updated_at = ? "
            "WHERE session_id = ? AND state IN ('collecting', 'proposing')",
            (json.dumps(scenario, ensure_ascii=False), now, session_id),
        )
        await self._db.commit()
        return await self.get(session_id)

    async def set_characters(
        self, session_id: str, characters: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not isinstance(characters, list):
            return None
        now = time.time()
        await self._db.execute(
            "UPDATE rp_director_sessions "
            "SET characters_json = ?, state = 'proposing', updated_at = ? "
            "WHERE session_id = ? AND state IN ('collecting', 'proposing')",
            (json.dumps(characters, ensure_ascii=False), now, session_id),
        )
        await self._db.commit()
        return await self.get(session_id)

    async def mark_committed(self, session_id: str) -> None:
        await self._db.execute(
            "UPDATE rp_director_sessions "
            "SET state = 'committed', updated_at = ? "
            "WHERE session_id = ?",
            (time.time(), session_id),
        )
        await self._db.commit()

    async def mark_cancelled(self, session_id: str) -> None:
        await self._db.execute(
            "UPDATE rp_director_sessions "
            "SET state = 'cancelled', updated_at = ? "
            "WHERE session_id = ?",
            (time.time(), session_id),
        )
        await self._db.commit()

    async def list_active(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        async with self._db.execute(
            "SELECT session_id, state, scenario_json, characters_json, "
            "user_intent, started_at, updated_at "
            "FROM rp_director_sessions "
            "WHERE state IN ('collecting', 'proposing') "
            "ORDER BY updated_at DESC"
        ) as cur:
            async for row in cur:
                out.append(_row_to_dict(row))
        return out

    async def expire_idle(self, max_age_seconds: float) -> list[str]:
        """Auto-cancel sessions idle longer than ``max_age_seconds``.

        Returns the list of session ids that were just cancelled.
        """
        if max_age_seconds <= 0:
            return []
        cutoff = time.time() - max_age_seconds
        ids: list[str] = []
        async with self._db.execute(
            "SELECT session_id FROM rp_director_sessions "
            "WHERE state IN ('collecting', 'proposing') AND updated_at < ?",
            (cutoff,),
        ) as cur:
            async for row in cur:
                ids.append(row[0])
        if not ids:
            return []
        await self._db.execute(
            "UPDATE rp_director_sessions "
            "SET state = 'cancelled', updated_at = ? "
            "WHERE state IN ('collecting', 'proposing') AND updated_at < ?",
            (time.time(), cutoff),
        )
        await self._db.commit()
        return ids


def _row_to_dict(row: Any) -> dict[str, Any]:
    scenario = _safe_json_obj(row[2])
    characters = _safe_json_list(row[3])
    return {
        "session_id": row[0],
        "state": row[1],
        "scenario": scenario,
        "characters": characters,
        "user_intent": row[4] or "",
        "started_at": float(row[5] or 0.0),
        "updated_at": float(row[6] or 0.0),
    }


def _safe_json_obj(raw: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_json_list(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]
