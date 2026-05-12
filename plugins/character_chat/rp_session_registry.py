"""
Registry of all roleplay session containers (Phase 13).

The character_chat plugin owns one :class:`RPSessionRegistry`
which is the gateway to every :class:`RPSessionContainer`. The
registry:

* discovers existing session folders under ``data/rp_sessions/``
* lazily opens containers (we don't pre-open SQLite for every
  historic session at startup)
* serialises ``get_or_create`` per session_id so two concurrent
  attaches can't both try to create the same folder
* closes all open container handles on plugin shutdown

The character_chat plugin treats the registry as the single
truth for "is this an RP session?" — anything outside the
``rp_sessions`` folder is, by definition, not RP-isolated.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from .rp_session_store import MemoryBackend, RPSessionContainer

if TYPE_CHECKING:  # pragma: no cover
    pass

log = structlog.get_logger(__name__)


class RPSessionRegistry:
    """Manages every RP session container under ``root``.

    Parameters
    ----------
    root
        Directory under which session folders live. Created on first
        use if missing.
    memory
        Memory backend (:class:`MemoryBackend`) shared by every
        container — usually the application's :class:`MemoryManager`.
    """

    def __init__(self, root: Path, memory: MemoryBackend) -> None:
        self._root = Path(root)
        self._memory = memory
        # session_id → open container handle (lazy populated)
        self._cache: dict[str, RPSessionContainer] = {}
        # session_id → asyncio.Lock so create/get races don't double-open.
        self._locks: dict[str, asyncio.Lock] = {}
        # Coarse lock around lock-table mutation.
        self._meta_lock = asyncio.Lock()
        self._closed = False

    # ─── Disk-level introspection ────────────────────────────────────

    def is_rp_session(self, session_id: str) -> bool:
        """Cheap check: does a folder exist for this session_id?

        Does NOT open the container. Useful in hot-paths like recall
        routing where we only want to know "should I take the RP
        codepath?".
        """
        if not session_id:
            return False
        return (self._root / session_id).is_dir()

    async def list_session_ids(self) -> list[str]:
        """All session_ids that have a folder on disk."""
        if not self._root.exists():
            return []
        return sorted(
            p.name for p in self._root.iterdir() if p.is_dir()
        )

    # ─── Container access ────────────────────────────────────────────

    async def get(self, session_id: str) -> RPSessionContainer | None:
        """Open and return an existing container, or ``None``."""
        if not self.is_rp_session(session_id):
            return None
        lock = await self._lock_for(session_id)
        async with lock:
            ct = self._cache.get(session_id)
            if ct is not None:
                return ct
            ct = await RPSessionContainer.open(
                self._root, session_id, self._memory
            )
            self._cache[session_id] = ct
            return ct

    async def get_or_create(
        self,
        session_id: str,
        *,
        title: str = "",
        scene: str = "",
        tracked_stats: dict[str, str] | None = None,
    ) -> RPSessionContainer:
        """Open existing or make fresh. Idempotent on re-call.

        On creation, ``title`` / ``scene`` / ``tracked_stats`` seed the
        new ``session.json``. On re-open of an existing container, those
        params are **ignored** — to update them, call
        :meth:`RPSessionContainer.update_meta` on the returned handle.
        """
        lock = await self._lock_for(session_id)
        async with lock:
            ct = self._cache.get(session_id)
            if ct is not None:
                return ct
            if (self._root / session_id).is_dir():
                ct = await RPSessionContainer.open(
                    self._root, session_id, self._memory
                )
            else:
                self._root.mkdir(parents=True, exist_ok=True)
                ct = await RPSessionContainer.create(
                    self._root,
                    session_id,
                    self._memory,
                    title=title,
                    scene=scene,
                    tracked_stats=tracked_stats,
                )
            self._cache[session_id] = ct
            return ct

    async def destroy(self, session_id: str) -> bool:
        """Permanently delete a session. Returns ``True`` if it existed."""
        lock = await self._lock_for(session_id)
        async with lock:
            ct = self._cache.pop(session_id, None)
            if ct is None:
                if not self.is_rp_session(session_id):
                    return False
                ct = await RPSessionContainer.open(
                    self._root, session_id, self._memory
                )
            await ct.destroy()
            return True

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Close every open container handle. Memory backend untouched."""
        if self._closed:
            return
        self._closed = True
        for session_id, ct in list(self._cache.items()):
            try:
                await ct.close()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "rp_registry.close_failed",
                    session_id=session_id,
                    error=str(exc),
                )
        self._cache.clear()

    # ─── Internal ────────────────────────────────────────────────────

    async def _lock_for(self, session_id: str) -> asyncio.Lock:
        async with self._meta_lock:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_id] = lock
            return lock


__all__ = ["RPSessionRegistry"]
