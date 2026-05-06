"""
Per-kind upload processors.

Each processor is a free async function with the same shape:

    async def process_<kind>(
        data: bytes, *, filename: str, mime: str, session_id: str,
        store: UploadStore, memory: MemoryManager | None,
        voice: VoiceManager | None,
    ) -> dict[str, Any]

The returned dict is what the frontend receives as the upload-complete
manifest. It always contains ``kind`` and the persisted ``upload_id``;
kind-specific extras (image dims, document chunk count, code language,
audio transcript) are added on top.

Memory persistence is best-effort: if ``memory`` is ``None`` (e.g. the
embedding client never initialised), the file still gets stored on
disk, the user just doesn't get RAG over it later.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from .parsers import parse_code, parse_html, parse_pdf, parse_text, detect_language
from .store import UploadRecord, UploadStore


log = logging.getLogger(__name__)


# Approximate words-per-token for budgeting truncation. Used only to cap
# absurdly large code/docs so a 10 MB log doesn't blow up the memory
# embed batch. Real token counting happens elsewhere.
_CHARS_PER_TOKEN = 4


# ─── Image ─────────────────────────────────────────────────────────────


async def process_image(
    data: bytes,
    *,
    filename: str,
    mime: str,
    session_id: str,
    store: UploadStore,
    memory: Any = None,
    voice: Any = None,
) -> dict[str, Any]:
    """Persist an image and return the manifest the chat can attach.

    The returned ``data_url`` is a complete ``data:image/...;base64,...``
    string ready to drop into a chat ``image_url`` content block. We
    keep the original bytes on disk too so the bubble can render via
    the ``/uploads/...`` static URL instead of bloating the history JSON.
    """
    width, height, normalised, normalised_mime, normalised_ext = _normalise_image(data, mime)
    record = await store.write(
        session_id=session_id,
        kind="image",
        filename=filename,
        ext=normalised_ext,
        mime=normalised_mime,
        data=normalised,
    )
    b64 = base64.b64encode(normalised).decode("ascii")
    data_url = f"data:{normalised_mime};base64,{b64}"
    return {
        "ok": True,
        "kind": "image",
        "upload_id": record.upload_id,
        "url": store.url_for(record),
        "filename": record.filename,
        "size": record.size,
        "mime": normalised_mime,
        "width": width,
        "height": height,
        # data_url is heavy — only included so the frontend can pass it
        # straight back as part of the next chat message's `attachments`.
        "data_url": data_url,
    }


def _normalise_image(data: bytes, mime: str) -> tuple[int, int, bytes, str, str]:
    """Open the image with Pillow, return (w, h, bytes, mime, ext).

    If decoding fails we keep the original bytes — uploads stay usable
    even when Pillow can't parse them. The LLM may still cope (llama.cpp
    does its own decoding via stb_image).
    """
    width = 0
    height = 0
    out_bytes = data
    out_mime = mime or "application/octet-stream"
    out_ext = _ext_from_mime(out_mime) or "bin"
    try:
        from PIL import Image  # local import — avoids hard dep on PIL for tests
    except ImportError:
        return width, height, out_bytes, out_mime, out_ext

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            width, height = img.size
            fmt = (img.format or "").lower()
            # Map Pillow format → MIME/ext. Pillow normalises JPEG/PNG/WEBP/GIF.
            fmt_map = {
                "jpeg": ("image/jpeg", "jpg"),
                "png": ("image/png", "png"),
                "webp": ("image/webp", "webp"),
                "gif": ("image/gif", "gif"),
            }
            if fmt in fmt_map:
                out_mime, out_ext = fmt_map[fmt]
    except Exception:
        # Either not a known image format or corrupt — keep originals.
        pass
    return width, height, out_bytes, out_mime, out_ext


def _ext_from_mime(mime: str) -> str | None:
    if not mime:
        return None
    return {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "application/pdf": "pdf",
        "text/plain": "txt",
        "text/markdown": "md",
        "text/html": "html",
        "audio/wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp4": "m4a",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
    }.get(mime.lower())


# ─── Document ──────────────────────────────────────────────────────────


async def process_document(
    data: bytes,
    *,
    filename: str,
    mime: str,
    session_id: str,
    store: UploadStore,
    memory: Any = None,
    voice: Any = None,
) -> dict[str, Any]:
    """Parse a document, store it, and ingest into knowledge memory."""
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    text = ""
    if ext == "pdf" or mime == "application/pdf":
        text = parse_pdf(data)
        ext = "pdf"
    elif ext in ("html", "htm") or mime in ("text/html", "application/xhtml+xml"):
        text = parse_html(data)
        ext = "html" if ext != "htm" else "htm"
    else:
        # Generic text path. Catches .txt .md .rst .csv .log .json .yaml ...
        text = parse_text(data)
        if not ext:
            ext = "txt"

    excerpt = text[:400]
    record = await store.write(
        session_id=session_id,
        kind="document",
        filename=filename,
        ext=ext,
        mime=mime or "application/octet-stream",
        data=data,
        text_excerpt=excerpt,
    )

    chunks_indexed = 0
    if memory is not None and text.strip():
        chunks_indexed = await _ingest_to_memory(
            text=text,
            memory=memory,
            session_id=session_id,
            record=record,
            kind_in_memory="docs",
        )

    return {
        "ok": True,
        "kind": "document",
        "upload_id": record.upload_id,
        "url": store.url_for(record),
        "filename": record.filename,
        "size": record.size,
        "mime": record.mime,
        "chars": len(text),
        "chunks_indexed": chunks_indexed,
        "excerpt": excerpt,
    }


# ─── Code ──────────────────────────────────────────────────────────────


async def process_code(
    data: bytes,
    *,
    filename: str,
    mime: str,
    session_id: str,
    store: UploadStore,
    memory: Any = None,
    voice: Any = None,
) -> dict[str, Any]:
    """Persist a code file as inline live-context (no chunking).

    Code files are *not* chunked into the knowledge base by default —
    they're meant as fresh per-task context for coding sessions. The
    text excerpt + the static URL is enough; the agent can fetch the
    full content via a future ``read_upload`` tool when it actually
    needs the lines.
    """
    text, lang, lines = parse_code(data, filename=filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    excerpt = text[:400]

    record = await store.write(
        session_id=session_id,
        kind="code",
        filename=filename,
        ext=ext,
        mime=mime or "text/plain",
        data=data,
        text_excerpt=excerpt,
    )

    # Optional knowledge-store — code files are usually large but not
    # always. We index only files under ~20 KB so a 200 MB minified JS
    # bundle doesn't take down the embedding queue.
    chunks_indexed = 0
    if memory is not None and 0 < len(text) < 20_000:
        chunks_indexed = await _ingest_to_memory(
            text=text,
            memory=memory,
            session_id=session_id,
            record=record,
            kind_in_memory="code",
            extra_metadata={"language": lang},
        )

    return {
        "ok": True,
        "kind": "code",
        "upload_id": record.upload_id,
        "url": store.url_for(record),
        "filename": record.filename,
        "size": record.size,
        "mime": record.mime,
        "language": lang,
        "lines": lines,
        "chars": len(text),
        "chunks_indexed": chunks_indexed,
        "excerpt": excerpt,
    }


# ─── Audio ─────────────────────────────────────────────────────────────


async def process_audio(
    data: bytes,
    *,
    filename: str,
    mime: str,
    session_id: str,
    store: UploadStore,
    memory: Any = None,
    voice: Any = None,
) -> dict[str, Any]:
    """Persist + transcribe an audio file."""
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower() or "wav"
    record = await store.write(
        session_id=session_id,
        kind="audio",
        filename=filename,
        ext=ext,
        mime=mime or "audio/wav",
        data=data,
    )

    transcript = ""
    if voice is not None:
        try:
            transcript = (await voice.transcribe(data)) or ""
        except Exception as exc:  # noqa: BLE001
            log.warning("uploads.audio_transcribe_failed: %s", exc)

    return {
        "ok": True,
        "kind": "audio",
        "upload_id": record.upload_id,
        "url": store.url_for(record),
        "filename": record.filename,
        "size": record.size,
        "mime": record.mime,
        "transcript": transcript,
    }


# ─── Memory ingestion helper ───────────────────────────────────────────


async def _ingest_to_memory(
    *,
    text: str,
    memory: Any,
    session_id: str,
    record: UploadRecord,
    kind_in_memory: str,
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    """Chunk + memory_store the text. Returns number of chunks stored.

    Tries the existing knowledge_acquisition chunker first; falls back
    to a naive paragraph-split. Failures are swallowed (logged) so a
    flaky embedding backend never blocks the upload itself.
    """
    chunks = _chunk_for_memory(text)
    if not chunks:
        return 0

    base_meta: dict[str, Any] = {
        "source": "uploads",
        "kind": kind_in_memory,
        "upload_id": record.upload_id,
        "filename": record.filename,
        "session_id": session_id,
    }
    if extra_metadata:
        base_meta.update(extra_metadata)

    stored = 0
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        meta = dict(base_meta, chunk_index=i, chunk_total=len(chunks))
        try:
            await memory.store(
                text=chunk,
                collection="context",
                metadata=meta,
            )
            stored += 1
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "uploads.memory_store_failed upload=%s chunk=%d error=%s",
                record.upload_id, i, exc,
            )
    return stored


def _chunk_for_memory(text: str, *, target_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Simple character-window chunker with paragraph awareness.

    Not as smart as the knowledge_acquisition chunker — but self-contained
    so the uploads module doesn't acquire a hard dependency on a plugin.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]

    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + target_chars, n)
        # Prefer to break at the next paragraph or sentence boundary.
        if end < n:
            for sep in ("\n\n", "\n", ". "):
                idx = text.rfind(sep, start + target_chars // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        out.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]
