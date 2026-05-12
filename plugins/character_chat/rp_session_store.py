"""
Per-RP-session storage container (Phase 13).

Each ``RPSessionContainer`` owns ONE folder under
``data/rp_sessions/<session_id>/`` plus ONE dedicated ChromaDB
collection (``rp__<session_id>``). It contains everything that
makes a roleplay session reproducible:

* ``session.json`` — header (title, scene, tracked_stats, created_at)
* ``state.json``   — per-character live state, ``{char_id: {key: value, …}}``
* ``messages.json``— user/assistant messages for resume / display
* ``turns.db``     — SQLite with this session's character turns
* the Chroma collection on the existing memory server

Mike's invariant from Phase 13:
    *„jede Session ist ein Ordner, in diesen Ordner wird alles
    gespeichert … wenn ich einen neuen Chat anfange kann ich so
    sicher sein das der memory wirklich leer ist!"*

So ``create()`` always yields an empty namespace, ``destroy()`` wipes
folder + Chroma collection, and recall is scoped to the per-session
collection — there is **no shared character_chat memory across
RP sessions**.

The container is plugin-internal. The character_chat plugin owns one
:class:`RPSessionRegistry` (see ``rp_session_registry.py``) which
manages the lifecycle of all containers.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


# ─── Memory backend protocol ────────────────────────────────────────


class MemoryBackend(Protocol):
    """Surface of :class:`MemoryManager` that the container uses.

    Defined as a Protocol so tests can pass an in-memory fake without
    spinning up a real ChromaDB. The real implementation is
    ``lexy_core.memory.memory_manager.MemoryManager`` plus the new
    ``ensure_collection`` / ``delete_collection`` methods.
    """

    async def ensure_collection(self, name: str) -> None: ...
    async def delete_collection(self, name: str) -> None: ...
    async def store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
    ) -> str: ...
    async def recall(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 5,
        project_id: str | None = None,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


# ─── Domain types ────────────────────────────────────────────────────


@dataclass
class TurnRow:
    """One character turn as persisted in the per-session ``turns.db``.

    Mirrors the legacy ``character_turns`` table minus the
    ``session_id`` column (the file IS the session).
    """

    id: str
    character_id: str
    character_name: str
    round_id: str
    order_num: int
    content: str
    skipped: bool = False
    trigger_kind: str = "user"
    trigger_text: str = ""
    created_at: float = field(default_factory=time.time)


# ─── Helpers ─────────────────────────────────────────────────────────


def _normalise_stat_key(raw: str) -> str:
    """Normalise a user-typed stat key to ``snake_case``.

    ``Clothing`` → ``clothing``, ``"Hunger Level"`` → ``hunger_level``.
    Whitespace and punctuation collapse to underscores. Empty input
    returns an empty string (caller drops it).
    """
    s = raw.strip().lower()
    if not s:
        return ""
    out: list[str] = []
    prev_under = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_under = False
        else:
            if not prev_under and out:
                out.append("_")
                prev_under = True
    return "".join(out).strip("_")


def parse_stats_input(raw: str) -> dict[str, str]:
    """Parse Mike's session-modal stats input into ``{key: default}``.

    Format: semicolon (or newline) separated ``key=value`` pairs.
    ``key`` alone (no ``=``) means "track this stat, no default value".

    Examples
    --------
    >>> parse_stats_input("Clothing=nackt; Posture=stehend; Mood; Hunger")
    {'clothing': 'nackt', 'posture': 'stehend', 'mood': '', 'hunger': ''}
    >>> parse_stats_input("")
    {}
    """
    result: dict[str, str] = {}
    if not raw or not raw.strip():
        return result
    # Split on either semicolon or newline so Mike can paste a list.
    chunks = []
    for line in raw.replace("\n", ";").split(";"):
        s = line.strip()
        if s:
            chunks.append(s)
    for chunk in chunks:
        if "=" in chunk:
            key, _, val = chunk.partition("=")
        else:
            key, val = chunk, ""
        norm = _normalise_stat_key(key)
        if not norm:
            continue
        result[norm] = val.strip()
    return result


def serialise_stats(stats: dict[str, str]) -> str:
    """Inverse of :func:`parse_stats_input` — render for UI display."""
    parts: list[str] = []
    for k, v in stats.items():
        if v:
            parts.append(f"{k}={v}")
        else:
            parts.append(k)
    return "; ".join(parts)


# ─── Container ──────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    character_name TEXT NOT NULL,
    round_id TEXT NOT NULL,
    order_num INTEGER NOT NULL,
    content TEXT NOT NULL,
    skipped INTEGER NOT NULL DEFAULT 0,
    trigger_kind TEXT NOT NULL DEFAULT 'user',
    trigger_text TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_char ON turns(character_id, created_at);
CREATE INDEX IF NOT EXISTS idx_turns_round ON turns(round_id, order_num);
CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);
"""


class RPSessionContainer:
    """One self-contained roleplay session.

    Construct via :meth:`create` (fresh) or :meth:`open` (existing).
    Always close via :meth:`close` when releasing the handle, or call
    :meth:`destroy` to permanently delete the session.

    Parameters
    ----------
    root
        Parent directory under which session folders live, typically
        ``Path("data/rp_sessions")``.
    session_id
        Stable id used both as folder name and as the suffix of the
        Chroma collection (``rp__<session_id>``).
    memory
        Implementation of :class:`MemoryBackend` — usually the
        application's :class:`MemoryManager`.
    """

    SESSION_FILE = "session.json"
    STATE_FILE = "state.json"
    MESSAGES_FILE = "messages.json"
    TURNS_FILE = "turns.db"

    def __init__(
        self,
        root: Path,
        session_id: str,
        memory: MemoryBackend,
    ) -> None:
        self._root = Path(root)
        self._session_id = session_id
        self._memory = memory
        self._db: aiosqlite.Connection | None = None
        # Single lock guards the JSON files (state, messages, session).
        # The SQLite DB has its own internal locking via aiosqlite.
        self._lock = asyncio.Lock()

    # ─── Identity ────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def folder(self) -> Path:
        return self._root / self._session_id

    @property
    def collection(self) -> str:
        """Chroma collection name. Stable across reopens."""
        return f"rp__{self._session_id}"

    # ─── Lifecycle ───────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        root: Path,
        session_id: str,
        memory: MemoryBackend,
        *,
        title: str = "",
        scene: str = "",
        tracked_stats: dict[str, str] | None = None,
    ) -> "RPSessionContainer":
        """Make a fresh empty container.

        Raises FileExistsError if the folder already exists — call
        :meth:`open` instead in that case, or :meth:`destroy` first
        then re-create.
        """
        ct = cls(root, session_id, memory)
        ct.folder.mkdir(parents=True, exist_ok=False)
        meta = {
            "session_id": session_id,
            "title": (title or "").strip(),
            "scene": (scene or "").strip(),
            "tracked_stats": tracked_stats or {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        await ct._write_json_async(cls.SESSION_FILE, meta)
        await ct._write_json_async(cls.STATE_FILE, {})
        await ct._write_json_async(cls.MESSAGES_FILE, [])
        await ct._init_db()
        await memory.ensure_collection(ct.collection)
        log.info(
            "rp_session.created",
            session_id=session_id,
            folder=str(ct.folder),
            collection=ct.collection,
            stats=list((tracked_stats or {}).keys()),
        )
        return ct

    @classmethod
    async def open(
        cls,
        root: Path,
        session_id: str,
        memory: MemoryBackend,
    ) -> "RPSessionContainer":
        """Re-open an existing container.

        The Chroma collection is ``ensure``d (idempotent) so a manual
        Chroma wipe on disk doesn't break the next recall.
        """
        ct = cls(root, session_id, memory)
        if not ct.folder.exists():
            raise FileNotFoundError(
                f"RP session folder missing: {ct.folder}"
            )
        # Backfill any missing files (forward-compat). A session created
        # before some new file existed should still open cleanly.
        for fn, default in (
            (cls.STATE_FILE, {}),
            (cls.MESSAGES_FILE, []),
        ):
            if not (ct.folder / fn).exists():
                await ct._write_json_async(fn, default)
        await ct._init_db()
        await memory.ensure_collection(ct.collection)
        log.debug(
            "rp_session.opened",
            session_id=session_id,
            collection=ct.collection,
        )
        return ct

    async def close(self) -> None:
        """Release the SQLite handle. Memory backend is shared — left alone."""
        if self._db is not None:
            try:
                await self._db.close()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "rp_session.close_failed",
                    session_id=self._session_id,
                    error=str(exc),
                )
            self._db = None

    async def destroy(self) -> None:
        """Permanently delete the session: folder + Chroma collection.

        After this call the container handle must not be used.
        """
        await self.close()
        try:
            await self._memory.delete_collection(self.collection)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "rp_session.delete_collection_failed",
                session_id=self._session_id,
                collection=self.collection,
                error=str(exc),
            )
        if self.folder.exists():
            try:
                shutil.rmtree(self.folder)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "rp_session.rmtree_failed",
                    session_id=self._session_id,
                    folder=str(self.folder),
                    error=str(exc),
                )
                raise
        log.info(
            "rp_session.destroyed",
            session_id=self._session_id,
            collection=self.collection,
        )

    # ─── Session metadata ────────────────────────────────────────────

    async def get_meta(self) -> dict[str, Any]:
        """Return ``session.json`` contents."""
        return await self._read_json_async(self.SESSION_FILE, {})

    async def update_meta(self, **fields: Any) -> dict[str, Any]:
        """Patch top-level keys in ``session.json``. Returns the new meta."""
        async with self._lock:
            meta = await self._read_json_sync_locked(self.SESSION_FILE, {})
            meta.update(fields)
            meta["updated_at"] = time.time()
            await self._write_json_sync_locked(self.SESSION_FILE, meta)
            return meta

    async def get_tracked_stats(self) -> dict[str, str]:
        meta = await self.get_meta()
        stats = meta.get("tracked_stats") or {}
        if not isinstance(stats, dict):
            return {}
        return dict(stats)

    async def set_tracked_stats(self, stats: dict[str, str]) -> None:
        await self.update_meta(tracked_stats=dict(stats))

    # ─── Per-character live state ────────────────────────────────────

    async def get_char_state(self, character_id: str) -> dict[str, str]:
        """Return this character's live state, or empty dict if none."""
        all_states = await self._read_json_async(self.STATE_FILE, {})
        st = all_states.get(character_id) or {}
        if not isinstance(st, dict):
            return {}
        return {str(k): str(v) for k, v in st.items()}

    async def set_char_state(
        self, character_id: str, state: dict[str, str]
    ) -> None:
        """Replace this character's live state with ``state``."""
        async with self._lock:
            all_states = await self._read_json_sync_locked(self.STATE_FILE, {})
            if state:
                all_states[character_id] = {
                    str(k): str(v) for k, v in state.items()
                }
            else:
                all_states.pop(character_id, None)
            await self._write_json_sync_locked(self.STATE_FILE, all_states)

    async def update_char_state(
        self, character_id: str, partial: dict[str, str]
    ) -> dict[str, str]:
        """Merge ``partial`` into the character's state.

        Only keys that appear in the session's ``tracked_stats`` are
        applied — anything else is silently dropped (Mike's
        configuration is the source of truth for what's relevant).
        Empty-string values **clear** the key.
        Returns the resulting state dict.
        """
        meta = await self.get_meta()
        allowed = set((meta.get("tracked_stats") or {}).keys())
        async with self._lock:
            all_states = await self._read_json_sync_locked(self.STATE_FILE, {})
            current = all_states.get(character_id) or {}
            for raw_key, raw_val in partial.items():
                key = _normalise_stat_key(raw_key)
                if not key or (allowed and key not in allowed):
                    continue
                val = str(raw_val).strip()
                if val == "":
                    current.pop(key, None)
                else:
                    current[key] = val
            if current:
                all_states[character_id] = current
            else:
                all_states.pop(character_id, None)
            await self._write_json_sync_locked(self.STATE_FILE, all_states)
            return dict(current)

    async def snapshot_template_for_char(
        self,
        character_id: str,
        template: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Initialise a character's state from the session's tracked-stats
        defaults (overridden by ``template`` if given).

        Called on ``character_attach`` so the LLM has a populated
        state-block on its very first turn — even though Mike's
        Phase 13 design no longer keeps a ``state`` column on the
        character itself, the session defaults usually carry the
        intent (e.g. ``clothing=nackt``).

        If the character already has a session-state, this is a no-op
        (we never clobber live state on re-attach).

        Returns the resulting state dict.
        """
        existing = await self.get_char_state(character_id)
        if existing:
            return existing
        defaults = await self.get_tracked_stats()
        merged: dict[str, str] = {k: v for k, v in defaults.items() if v}
        if template:
            for k, v in template.items():
                key = _normalise_stat_key(k)
                if not key or (defaults and key not in defaults):
                    continue
                if v:
                    merged[key] = str(v)
        if merged:
            await self.set_char_state(character_id, merged)
        return merged

    async def remove_char_state(self, character_id: str) -> None:
        """Drop this character's state entirely (used on detach)."""
        async with self._lock:
            all_states = await self._read_json_sync_locked(self.STATE_FILE, {})
            if all_states.pop(character_id, None) is not None:
                await self._write_json_sync_locked(
                    self.STATE_FILE, all_states
                )

    async def all_char_states(self) -> dict[str, dict[str, str]]:
        """Return ``{char_id: state_dict}`` for every char with state."""
        return await self._read_json_async(self.STATE_FILE, {})

    # ─── Character turns (SQLite) ────────────────────────────────────

    async def append_turn(self, turn: TurnRow) -> None:
        """Persist a single character turn. Caller supplies a stable id."""
        db = await self._ensure_db()
        await db.execute(
            "INSERT INTO turns (id, character_id, character_name, "
            "round_id, order_num, content, skipped, trigger_kind, "
            "trigger_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn.id,
                turn.character_id,
                turn.character_name,
                turn.round_id,
                turn.order_num,
                turn.content,
                1 if turn.skipped else 0,
                turn.trigger_kind,
                turn.trigger_text,
                turn.created_at,
            ),
        )
        await db.commit()

    async def list_turns(
        self,
        *,
        limit: int = 200,
        character_id: str | None = None,
    ) -> list[TurnRow]:
        """Chronological list of turns. Newest LAST (UI playback order)."""
        db = await self._ensure_db()
        if character_id:
            cursor = await db.execute(
                "SELECT id, character_id, character_name, round_id, "
                "order_num, content, skipped, trigger_kind, trigger_text, "
                "created_at FROM turns WHERE character_id = ? "
                "ORDER BY created_at ASC, order_num ASC LIMIT ?",
                (character_id, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT id, character_id, character_name, round_id, "
                "order_num, content, skipped, trigger_kind, trigger_text, "
                "created_at FROM turns "
                "ORDER BY created_at ASC, order_num ASC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            TurnRow(
                id=str(r[0]),
                character_id=str(r[1]),
                character_name=str(r[2]),
                round_id=str(r[3]),
                order_num=int(r[4]),
                content=str(r[5]),
                skipped=bool(r[6]),
                trigger_kind=str(r[7]),
                trigger_text=str(r[8]),
                created_at=float(r[9]),
            )
            for r in rows
        ]

    async def get_turn(self, turn_id: str) -> TurnRow | None:
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT id, character_id, character_name, round_id, "
            "order_num, content, skipped, trigger_kind, trigger_text, "
            "created_at FROM turns WHERE id = ?",
            (turn_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if not row:
            return None
        return TurnRow(
            id=str(row[0]),
            character_id=str(row[1]),
            character_name=str(row[2]),
            round_id=str(row[3]),
            order_num=int(row[4]),
            content=str(row[5]),
            skipped=bool(row[6]),
            trigger_kind=str(row[7]),
            trigger_text=str(row[8]),
            created_at=float(row[9]),
        )

    async def update_turn_content(self, turn_id: str, content: str) -> None:
        """Used by the regenerate flow — replaces a turn's text."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE turns SET content = ?, skipped = 0 WHERE id = ?",
            (content, turn_id),
        )
        await db.commit()

    async def delete_turn(self, turn_id: str) -> None:
        db = await self._ensure_db()
        await db.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
        await db.commit()

    async def list_turns_for_round(self, round_id: str) -> list[TurnRow]:
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT id, character_id, character_name, round_id, "
            "order_num, content, skipped, trigger_kind, trigger_text, "
            "created_at FROM turns WHERE round_id = ? "
            "ORDER BY order_num ASC",
            (round_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            TurnRow(
                id=str(r[0]),
                character_id=str(r[1]),
                character_name=str(r[2]),
                round_id=str(r[3]),
                order_num=int(r[4]),
                content=str(r[5]),
                skipped=bool(r[6]),
                trigger_kind=str(r[7]),
                trigger_text=str(r[8]),
                created_at=float(r[9]),
            )
            for r in rows
        ]

    # ─── Messages (user/assistant) ───────────────────────────────────

    async def append_message(self, msg: dict[str, Any]) -> None:
        """Append a user/assistant message (replaces sessions.json content for RP)."""
        async with self._lock:
            messages = await self._read_json_sync_locked(self.MESSAGES_FILE, [])
            if not isinstance(messages, list):
                messages = []
            messages.append(dict(msg))
            await self._write_json_sync_locked(self.MESSAGES_FILE, messages)

    async def list_messages(self, *, limit: int = 200) -> list[dict[str, Any]]:
        messages = await self._read_json_async(self.MESSAGES_FILE, [])
        if not isinstance(messages, list):
            return []
        if len(messages) > limit:
            return messages[-limit:]
        return list(messages)

    async def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        async with self._lock:
            await self._write_json_sync_locked(
                self.MESSAGES_FILE, list(messages)
            )

    # ─── Memory (Chroma + FTS via MemoryManager) ─────────────────────

    async def memory_write(
        self,
        *,
        text: str,
        character_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a memory item in this session's collection.

        ``character_id`` is added to the metadata so multi-character
        sessions can still filter recall to one speaker (Lena's
        memories don't appear in Sandra's recall — even though
        they're in the same collection).
        """
        meta = dict(metadata or {})
        meta["character_id"] = character_id
        meta.setdefault("session_id", self._session_id)
        meta.setdefault("source", "character_chat")
        return await self._memory.store(
            text=text,
            collection=self.collection,
            metadata=meta,
        )

    async def memory_recall(
        self,
        *,
        query: str,
        character_id: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Recall scoped to this session AND this character.

        Cross-session leak (Mike's Phase 13 trigger bug) is impossible
        here because the collection itself only holds this session's
        items.
        """
        return await self._memory.recall(
            query=query,
            collection=self.collection,
            limit=limit,
            metadata_equals={"character_id": character_id},
        )

    # ─── Internal helpers ────────────────────────────────────────────

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self._init_db()
        assert self._db is not None
        return self._db

    async def _init_db(self) -> None:
        db_path = self.folder / self.TURNS_FILE
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(db_path))
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def _write_json_async(self, name: str, data: Any) -> None:
        async with self._lock:
            await self._write_json_sync_locked(name, data)

    async def _read_json_async(self, name: str, default: Any) -> Any:
        async with self._lock:
            return await self._read_json_sync_locked(name, default)

    async def _write_json_sync_locked(self, name: str, data: Any) -> None:
        path = self.folder / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Run blocking IO off the event loop so concurrent containers
        # don't serialise on each other's writes.
        await asyncio.to_thread(_atomic_write_json, tmp, path, data)

    async def _read_json_sync_locked(self, name: str, default: Any) -> Any:
        path = self.folder / name
        if not path.exists():
            return default if not isinstance(default, (dict, list)) else type(default)()
        return await asyncio.to_thread(_read_json, path, default)

    # ─── Debug ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RPSessionContainer(session_id={self._session_id!r}, "
            f"folder={self.folder}, collection={self.collection!r})"
        )


# ─── Module-level IO helpers (off-loop) ─────────────────────────────


def _atomic_write_json(tmp: Path, target: Path, data: Any) -> None:
    """Write JSON to a temp file then rename — crash-safe."""
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp.write_text(payload, encoding="utf-8")
    # On Windows, rename across existing target needs replace().
    tmp.replace(target)


def _read_json(path: Path, default: Any) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default if not isinstance(default, (dict, list)) else type(default)()
    if not raw.strip():
        return default if not isinstance(default, (dict, list)) else type(default)()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning(
            "rp_session.json_decode_failed",
            path=str(path),
            error=str(exc),
        )
        return default if not isinstance(default, (dict, list)) else type(default)()


__all__ = [
    "MemoryBackend",
    "TurnRow",
    "RPSessionContainer",
    "parse_stats_input",
    "serialise_stats",
]
