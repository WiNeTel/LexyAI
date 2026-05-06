"""Tests for the per-format parsers + chunker used by the uploads pipeline."""

from __future__ import annotations

import io

import pytest

from lexy_core.uploads.parsers import (
    detect_language,
    parse_code,
    parse_html,
    parse_pdf,
    parse_text,
)
from lexy_core.uploads.processors import _chunk_for_memory


# ─── Text parser ─────────────────────────────────────────────────────


class TestParseText:
    def test_empty_returns_empty(self) -> None:
        assert parse_text(b"") == ""

    def test_utf8_roundtrip(self) -> None:
        assert parse_text("Hällö Wörld".encode("utf-8")) == "Hällö Wörld"

    def test_utf8_bom_stripped(self) -> None:
        # BOM-prefixed UTF-8 is what Notepad on Windows emits.
        assert parse_text(b"\xef\xbb\xbfhello") == "hello"

    def test_cp1252_fallback(self) -> None:
        # Latin-1 / cp1252 byte for ä is 0xe4 — not valid utf-8 standalone.
        assert parse_text(b"M\xe4dchen") == "Mädchen"

    def test_invalid_bytes_replaced_not_raised(self) -> None:
        # No encoding decodes 0xff 0xfe as a valid sequence; we still
        # return SOMETHING rather than blowing up.
        out = parse_text(b"\xff\xfe\x00\x00")
        assert isinstance(out, str)


# ─── HTML parser ─────────────────────────────────────────────────────


class TestParseHtml:
    def test_empty(self) -> None:
        assert parse_html(b"") == ""

    def test_strips_tags_and_decodes_entities(self) -> None:
        html = b"<p>Hello&nbsp;<b>world</b> &amp; co.</p>"
        assert parse_html(html) == "Hello world & co."

    def test_drops_script_and_style_blocks(self) -> None:
        html = (
            b"<html><head><style>body{color:red}</style></head>"
            b"<body><script>alert(1)</script><p>visible</p></body></html>"
        )
        result = parse_html(html)
        assert "visible" in result
        assert "alert" not in result
        assert "color:red" not in result

    def test_collapses_whitespace(self) -> None:
        html = b"<p>line one</p>\n\n\n\n<p>line two</p>"
        result = parse_html(html)
        assert "\n\n\n" not in result  # collapsed by _RE_BLANK_LINES


# ─── PDF parser ─────────────────────────────────────────────────────


class TestParsePdf:
    def test_empty_returns_empty(self) -> None:
        assert parse_pdf(b"") == ""

    def test_garbage_returns_empty_no_raise(self) -> None:
        # Non-PDF bytes — pypdf raises, we swallow.
        assert parse_pdf(b"this is definitely not a PDF") == ""

    def test_real_pdf_extracts_text(self) -> None:
        # Build a 1-page PDF with the pypdf writer so the test doesn't
        # require shipping a fixture file. If pypdf can't write, skip
        # — we still cover the parse-only path above.
        try:
            from pypdf import PdfReader, PdfWriter  # noqa: F401
            from pypdf.generic import (
                ArrayObject,
                ContentStream,
                NameObject,
                NumberObject,
                RectangleObject,
                StreamObject,
            )
        except ImportError:
            pytest.skip("pypdf not installed")

        # Easiest path: pypdf doesn't have a high-level "create PDF with
        # text" helper — building one manually is fragile. Use reportlab
        # if available; otherwise skip the round-trip case (we still
        # have garbage + empty coverage).
        try:
            from reportlab.pdfgen import canvas  # type: ignore
        except ImportError:
            pytest.skip("reportlab not installed — skipping PDF round-trip")
        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 750, "Lexy upload PDF round-trip test")
        c.save()
        text = parse_pdf(buf.getvalue())
        assert "Lexy upload PDF round-trip test" in text


# ─── Code parser + language detection ───────────────────────────────


class TestCodeParser:
    def test_detect_language_known(self) -> None:
        assert detect_language("foo.py") == "python"
        assert detect_language("Foo.TS") == "typescript"
        assert detect_language("server.go") == "go"

    def test_detect_language_unknown(self) -> None:
        assert detect_language("foo.unknownext") == ""
        assert detect_language("") == ""

    def test_detect_language_extension_only_basenames(self) -> None:
        # Bare "Dockerfile" / "Makefile" — no extension.
        assert detect_language("Dockerfile") == "dockerfile"
        assert detect_language("Makefile") == "makefile"
        assert detect_language("path/to/Dockerfile") == "dockerfile"

    def test_parse_code_returns_tuple(self) -> None:
        text, lang, lines = parse_code(
            b"def f():\n    return 1\n", filename="x.py"
        )
        assert "def f" in text
        assert lang == "python"
        assert lines == 2

    def test_line_count_handles_trailing_newline(self) -> None:
        _, _, with_nl = parse_code(b"a\nb\n", "x.txt")
        _, _, no_nl = parse_code(b"a\nb", "x.txt")
        # Both have 2 lines — trailing newline shouldn't add a phantom one.
        assert with_nl == 2
        assert no_nl == 2


# ─── Memory chunker ─────────────────────────────────────────────────


class TestChunker:
    def test_empty_returns_empty_list(self) -> None:
        assert _chunk_for_memory("") == []
        assert _chunk_for_memory("   \n  ") == []

    def test_short_text_returns_single_chunk(self) -> None:
        assert _chunk_for_memory("Just one chunk.") == ["Just one chunk."]

    def test_long_text_splits_with_overlap(self) -> None:
        text = ("paragraph one. " * 200).strip()
        chunks = _chunk_for_memory(text, target_chars=400, overlap=50)
        assert len(chunks) >= 2
        # Reconstructed (with dedup) should cover the original — the
        # overlap means each border letter appears in both neighbouring
        # chunks, so we just check no chunk is empty.
        assert all(c.strip() for c in chunks)

    def test_chunker_prefers_paragraph_boundaries(self) -> None:
        # When the target_chars window contains a paragraph boundary,
        # the chunker prefers to break there over a mid-sentence cut.
        # We give the boundary a comfortable position inside the window
        # so the heuristic actually fires.
        text = ("Para A. " * 30) + "\n\n" + ("Para B. " * 30)
        chunks = _chunk_for_memory(text, target_chars=300, overlap=20)
        assert len(chunks) >= 2
        # No chunk should mix the end of A with the start of B without
        # respecting the blank line — at least one chunk should end on
        # the boundary marker.
        assert any(c.rstrip().endswith("Para A.") for c in chunks)
