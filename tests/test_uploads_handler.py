"""Tests for the UploadHandler validation + dispatch logic.

The handler is the surface the gateway routes call. Most of the real
work happens in the per-kind processors (covered separately), so these
tests focus on:

* Validation: size limits, MIME prefixes, extension whitelists.
* Dispatch: the right processor runs for each handle_* method.
* Persistence: an UploadStore row + on-disk file land where expected.
* Error mapping: bad inputs raise UploadValidationError, not a generic
  500 — the gateway maps that to a 400 response.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from lexy_core.uploads.handler import (
    DEFAULT_LIMITS,
    UploadHandler,
    UploadValidationError,
)
from lexy_core.uploads.store import UploadStore


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> UploadStore:
    s = UploadStore(root=tmp_path / "uploads", db_path=tmp_path / "idx.db")
    await s.init()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def handler(store: UploadStore) -> UploadHandler:
    # Memory + voice are None — processors run their no-op paths.
    return UploadHandler(store=store, memory=None, voice=None)


# ─── Validation ──────────────────────────────────────────────────────


class TestUploadValidation:
    @pytest.mark.asyncio
    async def test_empty_upload_rejected(self, handler: UploadHandler) -> None:
        with pytest.raises(UploadValidationError, match="empty"):
            await handler.handle_image(
                data=b"",
                filename="x.png",
                mime="image/png",
                session_id="s",
            )

    @pytest.mark.asyncio
    async def test_image_too_large(self, handler: UploadHandler) -> None:
        big = b"\x00" * (DEFAULT_LIMITS["image"]["max_bytes"] + 1)
        with pytest.raises(UploadValidationError, match="too large"):
            await handler.handle_image(
                data=big, filename="x.png", mime="image/png", session_id="s",
            )

    @pytest.mark.asyncio
    async def test_image_extension_must_match_whitelist(
        self, handler: UploadHandler
    ) -> None:
        with pytest.raises(UploadValidationError, match="extension"):
            await handler.handle_image(
                data=b"\x89PNG\r\n",
                filename="evil.exe",
                mime="image/png",
                session_id="s",
            )

    @pytest.mark.asyncio
    async def test_image_mime_must_be_image(
        self, handler: UploadHandler
    ) -> None:
        with pytest.raises(UploadValidationError, match="MIME"):
            await handler.handle_image(
                data=b"\x89PNG\r\n",
                filename="x.png",
                mime="application/zip",
                session_id="s",
            )

    @pytest.mark.asyncio
    async def test_audio_mime_must_be_audio(
        self, handler: UploadHandler
    ) -> None:
        with pytest.raises(UploadValidationError, match="MIME"):
            await handler.handle_audio(
                data=b"RIFFXX",
                filename="x.wav",
                mime="image/png",
                session_id="s",
            )

    @pytest.mark.asyncio
    async def test_document_mime_lenient(
        self, handler: UploadHandler
    ) -> None:
        # docs accept text/* and application/* prefix; octet-stream is
        # tolerated for non-image/audio kinds (browsers often fail to
        # sniff). This call must NOT raise.
        result = await handler.handle_document(
            data=b"Hello world",
            filename="readme.md",
            mime="application/octet-stream",
            session_id="s",
        )
        assert result.ok is True
        assert result.kind == "document"

    @pytest.mark.asyncio
    async def test_code_extension_open(self, handler: UploadHandler) -> None:
        # Code uploads accept any extension (allowed_ext is empty set);
        # detection happens later via code_parser.
        result = await handler.handle_code(
            data=b"def foo(): pass\n",
            filename="weird.xyz",
            mime="text/plain",
            session_id="s",
        )
        assert result.ok is True
        assert result.kind == "code"


# ─── Dispatch + persistence ──────────────────────────────────────────


class TestHandlerDispatch:
    @pytest.mark.asyncio
    async def test_image_persists_file_and_index_row(
        self,
        handler: UploadHandler,
        store: UploadStore,
    ) -> None:
        # Tiny but valid PNG (1x1 transparent).
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9c"
            b"c\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = await handler.handle_image(
            data=png,
            filename="pixel.png",
            mime="image/png",
            session_id="sess-A",
        )
        assert result.ok is True
        payload = result.payload
        assert payload["kind"] == "image"
        assert payload["upload_id"] == result.upload_id
        # data_url is required so the chat can drop it straight into the
        # multimodal user message.
        assert payload["data_url"].startswith("data:image/")
        assert payload["url"].startswith("/uploads/sess-A/")
        # File and DB row exist.
        record = await store.get(result.upload_id)
        assert record is not None
        assert record.session_id == "sess-A"
        assert record.kind == "image"
        on_disk = store.file_path(record.session_id, record.upload_id, record.ext)
        assert on_disk.exists()
        assert on_disk.stat().st_size == len(png)

    @pytest.mark.asyncio
    async def test_document_returns_excerpt_and_chars(
        self, handler: UploadHandler, store: UploadStore
    ) -> None:
        text = "Title\n\nBody paragraph one. Body paragraph two."
        result = await handler.handle_document(
            data=text.encode("utf-8"),
            filename="notes.md",
            mime="text/markdown",
            session_id="s",
        )
        payload = result.payload
        assert payload["chars"] == len(text)
        assert payload["excerpt"].startswith("Title")
        # No memory wired → 0 chunks indexed.
        assert payload["chunks_indexed"] == 0

    @pytest.mark.asyncio
    async def test_code_detects_language(self, handler: UploadHandler) -> None:
        result = await handler.handle_code(
            data=b"const x = 1;\nfunction foo() { return x; }\n",
            filename="example.ts",
            mime="text/plain",
            session_id="s",
        )
        payload = result.payload
        assert payload["language"] == "typescript"
        assert payload["lines"] >= 2
        assert "const x = 1" in payload["excerpt"]

    @pytest.mark.asyncio
    async def test_audio_no_voice_no_transcript(
        self, handler: UploadHandler
    ) -> None:
        result = await handler.handle_audio(
            data=b"RIFFsmall_audio_blob",
            filename="memo.wav",
            mime="audio/wav",
            session_id="s",
        )
        payload = result.payload
        assert payload["kind"] == "audio"
        # Voice manager not provided in the fixture → empty transcript.
        assert payload["transcript"] == ""

    @pytest.mark.asyncio
    async def test_session_id_path_traversal_sanitised(
        self, handler: UploadHandler, store: UploadStore
    ) -> None:
        # If a malicious caller sneaks "../" into session_id, the store
        # must scrub it before resolving the on-disk path.
        result = await handler.handle_document(
            data=b"hi",
            filename="x.txt",
            mime="text/plain",
            session_id="../../../etc",
        )
        record = await store.get(result.upload_id)
        assert record is not None
        # Resolve the path and ensure it stays under the upload root.
        on_disk = store.file_path(
            record.session_id, record.upload_id, record.ext
        ).resolve()
        root = store.root.resolve()
        assert str(on_disk).startswith(str(root)), (on_disk, root)


# ─── Error path mapping ──────────────────────────────────────────────


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_validation_errors_subclass_value_error(
        self, handler: UploadHandler
    ) -> None:
        # The gateway uses ``except UploadValidationError`` to return 400.
        # If someone changes the parent class, this guard catches the
        # regression.
        try:
            await handler.handle_image(
                data=b"",
                filename="x.png",
                mime="image/png",
                session_id="s",
            )
        except UploadValidationError as exc:
            assert isinstance(exc, ValueError)
        else:
            pytest.fail("expected UploadValidationError")
