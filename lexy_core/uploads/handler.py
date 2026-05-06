"""
UploadHandler — top-level entry point for the four upload kinds.

The gateway calls one of:

    handler = UploadHandler.from_app(app)
    result  = await handler.handle_image(file, session_id=...)
    result  = await handler.handle_document(file, session_id=...)
    result  = await handler.handle_code(file, session_id=...)
    result  = await handler.handle_audio(file, session_id=...)

…and the handler does the size/MIME validation, hands off to the right
processor, and returns the manifest dict the frontend wraps into the
next chat message's ``attachments`` array.

Validation lives here (one place) so we can adjust limits in config
without touching the per-kind processors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .processors import (
    process_audio,
    process_code,
    process_document,
    process_image,
)
from .store import UploadStore


log = logging.getLogger(__name__)


UploadKind = str  # "image" | "document" | "code" | "audio"


@dataclass
class UploadResult:
    """Manifest returned to the frontend after a successful upload."""

    ok: bool
    kind: UploadKind
    upload_id: str
    payload: dict[str, Any]


# ─── Per-kind size + MIME limits ──────────────────────────────────────


# Defaults — kept conservative. Plugin/config can tighten later.
DEFAULT_LIMITS: dict[str, dict[str, Any]] = {
    "image": {
        "max_bytes": 8 * 1024 * 1024,    # 8 MB — covers phone photos
        "allowed_ext": {"jpg", "jpeg", "png", "webp", "gif"},
        "allowed_mime_prefix": ("image/",),
    },
    "document": {
        "max_bytes": 32 * 1024 * 1024,   # 32 MB — research PDFs can be big
        "allowed_ext": {
            "pdf", "txt", "md", "rst", "rtf", "csv", "tsv", "json", "yaml",
            "yml", "toml", "ini", "log", "html", "htm", "xml",
        },
        # Docs can come in with all sorts of MIME types; we don't
        # gatekeep too hard — extension match is the primary signal.
        "allowed_mime_prefix": (
            "application/", "text/",
        ),
    },
    "code": {
        "max_bytes": 4 * 1024 * 1024,    # 4 MB — way more than any sane source file
        # Empty set = accept anything (extension-driven detection in code_parser).
        "allowed_ext": set(),
        "allowed_mime_prefix": ("text/", "application/"),
    },
    "audio": {
        "max_bytes": 50 * 1024 * 1024,   # 50 MB — long voice memos
        "allowed_ext": {"wav", "mp3", "m4a", "ogg", "webm", "flac", "aac"},
        "allowed_mime_prefix": ("audio/",),
    },
}


class UploadValidationError(ValueError):
    """Raised when an upload fails size / type / mime validation."""


# ─── Handler ──────────────────────────────────────────────────────────


class UploadHandler:
    """Façade over the four processors + the store + the LLM/voice deps.

    The handler is created once per app via :meth:`from_app` and
    reused for every upload. Don't instantiate this directly in tests —
    use the constructor with explicit deps so the test can inject fakes.
    """

    def __init__(
        self,
        *,
        store: UploadStore,
        memory: Any = None,
        voice: Any = None,
        limits: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._voice = voice
        self._limits = dict(limits or DEFAULT_LIMITS)

    @classmethod
    async def from_app(cls, app: Any) -> "UploadHandler":
        """Build a handler bound to a running ``LexyApp``.

        Initialises the underlying ``UploadStore`` if it doesn't exist
        yet — idempotent so repeated calls are safe.
        """
        store = getattr(app, "_upload_store", None)
        if store is None:
            store = UploadStore()
            await store.init()
            app._upload_store = store
        return cls(
            store=store,
            memory=getattr(app, "memory", None),
            voice=getattr(app, "voice", None),
        )

    @property
    def store(self) -> UploadStore:
        return self._store

    # ─── Per-kind entry points ────────────────────────────────────────

    async def handle_image(
        self,
        *,
        data: bytes,
        filename: str,
        mime: str,
        session_id: str,
    ) -> UploadResult:
        self._validate("image", filename, mime, len(data))
        payload = await process_image(
            data,
            filename=filename,
            mime=mime,
            session_id=session_id,
            store=self._store,
            memory=self._memory,
            voice=self._voice,
        )
        return UploadResult(
            ok=True,
            kind="image",
            upload_id=payload["upload_id"],
            payload=payload,
        )

    async def handle_document(
        self,
        *,
        data: bytes,
        filename: str,
        mime: str,
        session_id: str,
    ) -> UploadResult:
        self._validate("document", filename, mime, len(data))
        payload = await process_document(
            data,
            filename=filename,
            mime=mime,
            session_id=session_id,
            store=self._store,
            memory=self._memory,
            voice=self._voice,
        )
        return UploadResult(
            ok=True,
            kind="document",
            upload_id=payload["upload_id"],
            payload=payload,
        )

    async def handle_code(
        self,
        *,
        data: bytes,
        filename: str,
        mime: str,
        session_id: str,
    ) -> UploadResult:
        self._validate("code", filename, mime, len(data))
        payload = await process_code(
            data,
            filename=filename,
            mime=mime,
            session_id=session_id,
            store=self._store,
            memory=self._memory,
            voice=self._voice,
        )
        return UploadResult(
            ok=True,
            kind="code",
            upload_id=payload["upload_id"],
            payload=payload,
        )

    async def handle_audio(
        self,
        *,
        data: bytes,
        filename: str,
        mime: str,
        session_id: str,
    ) -> UploadResult:
        self._validate("audio", filename, mime, len(data))
        payload = await process_audio(
            data,
            filename=filename,
            mime=mime,
            session_id=session_id,
            store=self._store,
            memory=self._memory,
            voice=self._voice,
        )
        return UploadResult(
            ok=True,
            kind="audio",
            upload_id=payload["upload_id"],
            payload=payload,
        )

    # ─── Validation ───────────────────────────────────────────────────

    def _validate(
        self, kind: UploadKind, filename: str, mime: str, size: int
    ) -> None:
        rule = self._limits.get(kind)
        if rule is None:
            raise UploadValidationError(f"unknown upload kind: {kind}")
        if size <= 0:
            raise UploadValidationError("empty upload")
        if size > int(rule["max_bytes"]):
            raise UploadValidationError(
                f"{kind} too large ({size} > {rule['max_bytes']} bytes)"
            )

        ext_rules = rule.get("allowed_ext") or set()
        if ext_rules:
            ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
            if ext not in ext_rules:
                raise UploadValidationError(
                    f"{kind} extension not allowed: {ext!r} "
                    f"(allowed: {sorted(ext_rules)})"
                )

        mime_prefixes = tuple(rule.get("allowed_mime_prefix") or ())
        if mime_prefixes and mime:
            mime_lower = mime.lower()
            if not any(mime_lower.startswith(p) for p in mime_prefixes):
                # Be lenient — many uploads come with octet-stream MIME
                # because the browser couldn't sniff a custom code file.
                # Only enforce strictly for image/audio where the LLM
                # actually needs the type.
                if kind in ("image", "audio"):
                    raise UploadValidationError(
                        f"{kind} MIME not allowed: {mime!r}"
                    )

        # OK — passed.
        return None
