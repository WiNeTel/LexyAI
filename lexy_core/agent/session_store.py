"""
Lexy AI - SessionStore (v2 format with metadata).

Bounded conversation history per session. LexyAgent reads the history in
``_plan()`` and injects it between the system prompt and the current user
turn so the LLM has conversational context.

The store is thread-safe and — when given a ``persistent_path`` — writes
every mutation to disk via an atomic replace so sessions survive across
server restarts.

On-disk format (v2)::

    {
        "version": 2,
        "saved_at": 1712851200.0,
        "max_messages": 20,
        "sessions": {
            "<session_id>": {
                "messages": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ],
                "meta": {
                    "project_id": "proj-abc" | null,
                    "created_at": 1712851200.0,
                    "updated_at": 1712851299.0,
                    "title": "Erste User-Message (max 60 chars)..." | null
                }
            }
        }
    }

Backward-compat: v1 format (``sessions[id]`` is a bare message list) is
auto-wrapped into ``{"messages": [...], "meta": {...}}`` on load, and
rewritten in v2 shape on the next save.

Plugins that need richer durable history (full audit log, timestamps,
attachments) can still subscribe to ``core.user_message`` and
``core.ai_response`` events and persist them separately.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import structlog

Role = Literal["system", "user", "assistant", "tool"]

log = structlog.get_logger(__name__)

_STORE_VERSION = 2
_TITLE_MAX_LEN = 60


class SessionStore:
    """
    Thread-safe ring buffer of messages per session with optional
    JSON-on-disk persistence.

    Each session is stored as an entry containing a ``messages`` list and
    a ``meta`` dict (project_id, created_at, updated_at, title).

    ``max_messages`` caps the number of *non-system* messages kept per
    session. When the limit is reached the oldest pair (user+assistant)
    is dropped so the window always starts on a user turn.

    If ``persistent_path`` is given, every mutation flushes the store
    atomically to that file. Missing or corrupt files are logged and
    treated as an empty store.
    """

    def __init__(
        self,
        max_messages: int = 20,
        persistent_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._max = max(2, int(max_messages))
        # Each value is {"messages": [...], "meta": {...}}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        # Resolve to absolute path so working-directory changes between runs
        # can't make sessions "disappear" by writing to a different location.
        self._path: Path | None = (
            Path(persistent_path).resolve() if persistent_path else None
        )
        if self._path is not None:
            log.info(
                "session_store.path",
                path=str(self._path),
                exists=self._path.exists(),
            )
            self.load()

    # ─── Persistence ────────────────────────────────────────────────

    @property
    def path(self) -> Path | None:
        return self._path

    def load(self) -> int:
        """
        Load sessions from ``self._path``. Returns the number of sessions
        restored. Supports both v1 (bare list) and v2 (dict with meta).
        Missing or unreadable files result in an empty store and a warning
        log, never an exception.
        """
        if self._path is None:
            return 0
        with self._lock:
            self._sessions = {}
            if not self._path.exists():
                return 0
            try:
                raw = self._path.read_text(encoding="utf-8")
                if not raw.strip():
                    return 0
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "session_store.load_failed",
                    path=str(self._path),
                    error=str(exc),
                )
                # Try loading the backup — one corrupted save shouldn't
                # erase the user's entire session history.
                bak = self._path.with_suffix(self._path.suffix + ".bak")
                if bak.exists():
                    log.info("session_store.trying_backup", path=str(bak))
                    try:
                        raw = bak.read_text(encoding="utf-8")
                        data = json.loads(raw)
                    except (OSError, json.JSONDecodeError) as bak_exc:
                        log.error(
                            "session_store.backup_also_failed",
                            error=str(bak_exc),
                        )
                        return 0
                else:
                    return 0

            sessions = data.get("sessions") if isinstance(data, dict) else None
            if not isinstance(sessions, dict):
                log.warning(
                    "session_store.load_invalid_shape",
                    path=str(self._path),
                )
                return 0

            file_version = data.get("version") if isinstance(data, dict) else 1
            restored = 0
            migrated_from_v1 = 0
            for session_id, raw_value in sessions.items():
                if not isinstance(session_id, str):
                    continue
                entry = self._normalize_entry(raw_value)
                if entry is None:
                    continue
                self._trim(entry["messages"])
                self._sessions[session_id] = entry
                restored += 1
                if isinstance(raw_value, list):
                    migrated_from_v1 += 1

            log.info(
                "session_store.loaded",
                path=str(self._path),
                sessions=restored,
                file_version=file_version,
                migrated_v1_entries=migrated_from_v1,
            )
            # Rewrite on-disk so subsequent reads see v2 shape
            if migrated_from_v1 > 0 or file_version != _STORE_VERSION:
                self._save_locked()
            return restored

    def _normalize_entry(
        self, raw_value: Any
    ) -> dict[str, Any] | None:
        """
        Convert a raw entry from disk into the canonical
        ``{"messages": [...], "meta": {...}}`` shape. Returns ``None``
        for values we cannot interpret.
        """
        # v1: bare list of messages
        if isinstance(raw_value, list):
            cleaned = self._clean_messages(raw_value)
            if not cleaned:
                return None
            return {
                "messages": cleaned,
                "meta": {
                    "project_id": None,
                    "created_at": 0.0,
                    "updated_at": 0.0,
                    "title": None,
                },
            }
        # v2: dict with messages + meta
        if isinstance(raw_value, dict):
            cleaned = self._clean_messages(raw_value.get("messages") or [])
            raw_meta = raw_value.get("meta") or {}
            if not isinstance(raw_meta, dict):
                raw_meta = {}
            meta: dict[str, Any] = {
                "project_id": raw_meta.get("project_id"),
                "created_at": float(raw_meta.get("created_at") or 0.0),
                "updated_at": float(raw_meta.get("updated_at") or 0.0),
                "title": raw_meta.get("title"),
            }
            # Allow empty-message sessions (registered but no first turn yet)
            return {"messages": cleaned, "meta": meta}
        return None

    @staticmethod
    def _clean_messages(raw: Any) -> list[dict[str, str]]:
        """Validate + shape raw messages into [{"role": ..., "content": ...}]."""
        cleaned: list[dict[str, str]] = []
        if not isinstance(raw, list):
            return cleaned
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            role = entry.get("role")
            content = entry.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    def save(self) -> bool:
        """
        Flush the current store to disk. Returns ``True`` on success,
        ``False`` if no path is configured or the write failed.
        """
        if self._path is None:
            return False
        with self._lock:
            return self._save_locked()

    def _save_locked(self) -> bool:
        """Inner writer — caller MUST hold ``self._lock``.

        Strategy: backup the current file BEFORE overwriting, then atomic
        temp+replace. If the main file is lost/corrupted, ``load()`` falls
        back to the ``.bak`` copy automatically.
        """
        assert self._path is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            # Backup the existing file before we overwrite. A single .bak
            # is enough — it survives one bad write. If even the backup is
            # corrupted, at least the user gets a warning at load time.
            if self._path.exists():
                bak = self._path.with_suffix(self._path.suffix + ".bak")
                try:
                    import shutil
                    shutil.copy2(str(self._path), str(bak))
                except OSError:
                    pass  # non-fatal; better to save than to abort on backup failure

            payload: dict[str, Any] = {
                "version": _STORE_VERSION,
                "saved_at": time.time(),
                "max_messages": self._max,
                "sessions": {
                    sid: {
                        "messages": [dict(msg) for msg in entry["messages"]],
                        "meta": dict(entry["meta"]),
                    }
                    for sid, entry in self._sessions.items()
                },
            }
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
            return True
        except OSError as exc:
            log.warning(
                "session_store.save_failed",
                path=str(self._path),
                error=str(exc),
            )
            return False

    def _persist(self) -> None:
        """Flush if a path is configured. Caller must hold the lock."""
        if self._path is not None:
            self._save_locked()

    # ─── Entry helpers ──────────────────────────────────────────────

    def _ensure_entry(
        self,
        session_id: str,
        project_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Get or create the session entry. Caller holds the lock."""
        entry = self._sessions.get(session_id)
        if entry is None:
            now = time.time()
            entry = {
                "messages": [],
                "meta": {
                    "project_id": project_id,
                    "created_at": now,
                    "updated_at": now,
                    "title": title,
                },
            }
            self._sessions[session_id] = entry
        return entry

    def _touch(self, entry: dict[str, Any]) -> None:
        """Update the ``updated_at`` timestamp. Caller holds the lock."""
        entry["meta"]["updated_at"] = time.time()

    # ─── Mutations ──────────────────────────────────────────────────

    def register_empty(
        self,
        session_id: str,
        project_id: str | None = None,
        title: str | None = None,
    ) -> bool:
        """
        Create an empty session slot with metadata. Idempotent: calling
        with an existing id returns ``False`` and leaves messages
        untouched (but updates project_id/title if they were ``None``
        before and a value is provided now).

        Returns ``True`` if a new slot was created, ``False`` otherwise.
        """
        if not session_id:
            return False
        with self._lock:
            existed = session_id in self._sessions
            entry = self._ensure_entry(session_id, project_id, title)
            if existed:
                # Enrich meta without clobbering existing non-None values
                meta = entry["meta"]
                if project_id is not None and meta.get("project_id") is None:
                    meta["project_id"] = project_id
                if title is not None and not meta.get("title"):
                    meta["title"] = title
                self._persist()
                return False
            self._persist()
            log.info(
                "session_store.registered",
                session_id=session_id,
                project_id=project_id,
            )
            return True

    def append(self, session_id: str, role: Role, content: str) -> None:
        """Append one message and enforce the window."""
        if not session_id:
            return
        with self._lock:
            entry = self._ensure_entry(session_id)
            entry["messages"].append({"role": role, "content": content})
            self._trim(entry["messages"])
            self._touch(entry)
            self._persist()

    def append_user(
        self,
        session_id: str,
        user_text: str,
        project_id: str | None = None,
    ) -> None:
        """
        Append a user turn immediately. Called from the agent BEFORE the
        LLM is invoked so the message survives a mid-stream crash.

        If the session is brand new AND the first user message has no
        title yet, we derive a short title from the text (first 60 chars,
        stripped).
        """
        if not session_id:
            return
        with self._lock:
            entry = self._ensure_entry(session_id, project_id=project_id)
            entry["messages"].append({"role": "user", "content": user_text})
            self._trim(entry["messages"])
            # Derive title from the first user message if not set
            if not entry["meta"].get("title"):
                title = " ".join(user_text.split())[:_TITLE_MAX_LEN].strip()
                if title:
                    entry["meta"]["title"] = title
            self._touch(entry)
            self._persist()

    def append_assistant(self, session_id: str, assistant_text: str) -> None:
        """
        Append an assistant turn. Called from the agent AFTER the LLM
        response finishes successfully. Safe to call even if no matching
        user turn exists (the window trim still works correctly).
        """
        if not session_id:
            return
        with self._lock:
            entry = self._ensure_entry(session_id)
            entry["messages"].append(
                {"role": "assistant", "content": assistant_text}
            )
            self._trim(entry["messages"])
            self._touch(entry)
            self._persist()

    def append_pair(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """
        Convenience: append both sides of a single turn at once.

        Kept for backwards compatibility with tests and callers that
        don't need crash-safe semantics. New production code should
        prefer ``append_user`` + ``append_assistant``.
        """
        if not session_id:
            return
        with self._lock:
            entry = self._ensure_entry(session_id)
            entry["messages"].append({"role": "user", "content": user_text})
            entry["messages"].append(
                {"role": "assistant", "content": assistant_text}
            )
            self._trim(entry["messages"])
            if not entry["meta"].get("title"):
                title = " ".join(user_text.split())[:_TITLE_MAX_LEN].strip()
                if title:
                    entry["meta"]["title"] = title
            self._touch(entry)
            self._persist()

    def _trim(self, history: list[dict[str, str]]) -> None:
        """Drop oldest entries until within the window, preserving pairs."""
        if len(history) <= self._max:
            return
        # Drop from the front; keep an even count so the next turn is a user
        # message (LLMs handle alternation better).
        drop = len(history) - self._max
        if drop % 2 == 1:
            drop += 1
        del history[:drop]

    def clear(self, session_id: str) -> int:
        """Remove a session's history. Returns the number of messages dropped."""
        with self._lock:
            entry = self._sessions.get(session_id)
            dropped = len(entry["messages"]) if entry else 0
            if session_id in self._sessions:
                self._sessions.pop(session_id, None)
                self._persist()
            return dropped

    def reset_all(self) -> None:
        """Drop every session."""
        with self._lock:
            self._sessions.clear()
            self._persist()

    # ─── Metadata mutations ─────────────────────────────────────────

    def set_project(
        self, session_id: str, project_id: str | None
    ) -> bool:
        """
        Set (or clear) the project a session belongs to. Returns ``True``
        if the session existed and the project was updated, ``False``
        otherwise.
        """
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            entry["meta"]["project_id"] = project_id
            self._touch(entry)
            self._persist()
            return True

    def set_title(self, session_id: str, title: str) -> bool:
        """Manually override a session's title."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return False
            entry["meta"]["title"] = title.strip()[:_TITLE_MAX_LEN] or None
            self._touch(entry)
            self._persist()
            return True

    # ─── Edit / pop helpers ─────────────────────────────────────────

    def pop_last_pair(
        self, session_id: str
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        """
        Remove the most recent (user, assistant) turn from a session.

        Returns the dropped ``(user_msg, assistant_msg)`` pair. Missing
        slots are ``None`` (e.g. if only the user message was stored).
        Used by the regenerate flow so the agent can re-run the same
        user turn without the stale assistant reply polluting history.
        """
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None or not entry["messages"]:
                return None, None
            history = entry["messages"]
            assistant_msg = None
            user_msg = None
            if history and history[-1]["role"] == "assistant":
                assistant_msg = history.pop()
            if history and history[-1]["role"] == "user":
                user_msg = history.pop()
            if assistant_msg is not None or user_msg is not None:
                self._touch(entry)
                self._persist()
            return user_msg, assistant_msg

    def replace_at(
        self, session_id: str, index: int, content: str
    ) -> dict[str, str] | None:
        """
        Replace the content of the message at ``index`` in the session's
        history. Returns the updated message, or ``None`` if the index
        is out of range.
        """
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            history = entry["messages"]
            if index < 0 or index >= len(history):
                return None
            history[index] = {**history[index], "content": content}
            self._touch(entry)
            self._persist()
            return dict(history[index])

    def delete_at(self, session_id: str, index: int) -> dict[str, str] | None:
        """
        Remove the message at ``index``. Returns the dropped message, or
        ``None`` if the index is out of range.
        """
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            history = entry["messages"]
            if index < 0 or index >= len(history):
                return None
            dropped = history.pop(index)
            self._touch(entry)
            self._persist()
            return dropped

    # ─── Reads ──────────────────────────────────────────────────────

    def get(self, session_id: str) -> list[dict[str, str]]:
        """Return a shallow copy of the session's messages."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return []
            return [dict(msg) for msg in entry["messages"]]

    def length(self, session_id: str) -> int:
        with self._lock:
            entry = self._sessions.get(session_id)
            return len(entry["messages"]) if entry else 0

    def sessions(self) -> list[str]:
        """List all known session ids, including empty ones."""
        with self._lock:
            return list(self._sessions.keys())

    def get_meta(self, session_id: str) -> dict[str, Any]:
        """Return a shallow copy of the session's metadata, or ``{}``."""
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                return {}
            return dict(entry["meta"])

    def sessions_with_meta(self) -> list[tuple[str, dict[str, Any], int]]:
        """
        Return ``(session_id, meta_dict, message_count)`` tuples for
        every known session. Used by the gateway listing endpoint.
        """
        with self._lock:
            return [
                (sid, dict(entry["meta"]), len(entry["messages"]))
                for sid, entry in self._sessions.items()
            ]
