"""
Upload-Store — file persistence + lightweight metadata index.

A single :class:`UploadStore` owns the directory ``data/uploads/`` and a
sidecar SQLite index. Every successful upload writes:

* the original bytes to ``data/uploads/<session_id>/<upload_id>.<ext>``
* one row in the ``uploads`` table:
  ``(upload_id, session_id, kind, filename, ext, mime, size, sha1,
    text_excerpt, created_at)``

The text-excerpt column lets us re-show ingested documents in the
sidebar without re-parsing the PDF every time the user scrolls. It also
makes garbage-collection trivial: drop rows older than N days, delete
the corresponding files.

The store does **not** do parsing or LLM calls — that's the
:mod:`processors` layer. It's a pure persistence helper.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import aiosqlite


# Where uploaded files live. Flat per-session subdirs keep the listing
# manageable and let us nuke a whole session's uploads with one rmtree.
UPLOADS_ROOT: Path = Path("data/uploads")

# Sidecar index — one DB shared by all sessions. Small, fast, easy to back up.
_DB_PATH: Path = UPLOADS_ROOT / "_index.db"

# Anything not in this set gets sanitised to ``.bin`` before storage.
# We intentionally allow more than the LLM cares about (e.g. .zip) so
# the store can also back generic file-attachments later. Validation of
# *what's actually allowed for which kind* lives in the handler.
_ALLOWED_EXT_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")


@dataclass
class UploadRecord:
    """One persisted upload."""

    upload_id: str
    session_id: str
    kind: str
    filename: str           # original client-supplied name
    ext: str                # sanitised extension (no dot)
    mime: str
    size: int
    sha1: str
    text_excerpt: str = ""  # first ~200 chars of extracted text, if any
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "filename": self.filename,
            "ext": self.ext,
            "mime": self.mime,
            "size": self.size,
            "sha1": self.sha1,
            "text_excerpt": self.text_excerpt,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "UploadRecord":
        return cls(
            upload_id=str(row["upload_id"]),
            session_id=str(row["session_id"]),
            kind=str(row["kind"]),
            filename=str(row.get("filename") or ""),
            ext=str(row.get("ext") or ""),
            mime=str(row.get("mime") or ""),
            size=int(row.get("size") or 0),
            sha1=str(row.get("sha1") or ""),
            text_excerpt=str(row.get("text_excerpt") or ""),
            created_at=float(row.get("created_at") or 0.0),
        )


class UploadStore:
    """Async wrapper around the on-disk + SQLite persistence."""

    def __init__(
        self,
        root: Path | str = UPLOADS_ROOT,
        db_path: Path | str | None = None,
    ) -> None:
        self._root = Path(root)
        self._db_path = Path(db_path) if db_path is not None else self._root / "_index.db"
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    async def init(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                upload_id      TEXT PRIMARY KEY,
                session_id     TEXT NOT NULL,
                kind           TEXT NOT NULL,
                filename       TEXT NOT NULL DEFAULT '',
                ext            TEXT NOT NULL DEFAULT '',
                mime           TEXT NOT NULL DEFAULT '',
                size           INTEGER NOT NULL DEFAULT 0,
                sha1           TEXT NOT NULL DEFAULT '',
                text_excerpt   TEXT NOT NULL DEFAULT '',
                created_at     REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_uploads_session ON uploads(session_id, created_at)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ─── Storage ──────────────────────────────────────────────────────

    @staticmethod
    def sanitise_ext(ext: str) -> str:
        ext = (ext or "").lstrip(".").lower()
        if not _ALLOWED_EXT_RE.match(ext):
            return "bin"
        return ext

    def file_path(self, session_id: str, upload_id: str, ext: str) -> Path:
        # Defensive: never let a caller-supplied session_id contain path
        # separators. ``data/uploads/<safe>/<id>.<ext>`` is the contract.
        safe_session = re.sub(r"[^A-Za-z0-9_\-]", "_", session_id) or "default"
        return self._root / safe_session / f"{upload_id}.{ext}"

    def url_for(self, record: UploadRecord) -> str:
        """Public URL under the ``/uploads`` static mount."""
        safe_session = re.sub(r"[^A-Za-z0-9_\-]", "_", record.session_id) or "default"
        return f"/uploads/{safe_session}/{record.upload_id}.{record.ext}"

    async def write(
        self,
        *,
        session_id: str,
        kind: str,
        filename: str,
        ext: str,
        mime: str,
        data: bytes,
        text_excerpt: str = "",
    ) -> UploadRecord:
        """Persist bytes + index row. Idempotent per content (sha1)."""
        if self._db is None:
            raise RuntimeError("UploadStore not initialised — call init() first")

        sha1 = hashlib.sha1(data).hexdigest()
        upload_id = uuid.uuid4().hex[:16]
        ext = self.sanitise_ext(ext)
        path = self.file_path(session_id, upload_id, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        record = UploadRecord(
            upload_id=upload_id,
            session_id=session_id,
            kind=kind,
            filename=filename or f"{upload_id}.{ext}",
            ext=ext,
            mime=mime or "application/octet-stream",
            size=len(data),
            sha1=sha1,
            text_excerpt=(text_excerpt or "")[:400],
        )

        row = record.to_row()
        cols = ",".join(row.keys())
        marks = ",".join("?" for _ in row)
        async with self._lock:
            await self._db.execute(
                f"INSERT INTO uploads ({cols}) VALUES ({marks})",
                tuple(row.values()),
            )
            await self._db.commit()
        return record

    async def get(self, upload_id: str) -> UploadRecord | None:
        if self._db is None:
            return None
        async with self._db.execute(
            "SELECT * FROM uploads WHERE upload_id = ?", (upload_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return UploadRecord.from_row(dict(row))

    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> list[UploadRecord]:
        if self._db is None:
            return []
        out: list[UploadRecord] = []
        async with self._db.execute(
            "SELECT * FROM uploads WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, max(1, int(limit))),
        ) as cur:
            async for row in cur:
                out.append(UploadRecord.from_row(dict(row)))
        return out

    async def delete(self, upload_id: str) -> bool:
        record = await self.get(upload_id)
        if record is None:
            return False
        path = self.file_path(record.session_id, record.upload_id, record.ext)
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        async with self._lock:
            assert self._db is not None
            await self._db.execute(
                "DELETE FROM uploads WHERE upload_id = ?", (upload_id,)
            )
            await self._db.commit()
        return True

    async def gc_older_than(self, max_age_seconds: float) -> int:
        """Garbage-collect uploads older than ``max_age_seconds``.

        Returns the number of files removed. Index rows are deleted too.
        Useful for the eventual cleanup task — not called automatically
        anywhere; left to a future scheduler entry.
        """
        if self._db is None:
            return 0
        cutoff = time.time() - max_age_seconds
        ids: list[str] = []
        async with self._db.execute(
            "SELECT upload_id FROM uploads WHERE created_at < ?",
            (cutoff,),
        ) as cur:
            async for row in cur:
                ids.append(str(row["upload_id"]))
        for uid in ids:
            await self.delete(uid)
        return len(ids)

    async def size_total(self) -> int:
        if self._db is None:
            return 0
        async with self._db.execute(
            "SELECT COALESCE(SUM(size), 0) AS total FROM uploads"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return 0
        return int(row["total"])
