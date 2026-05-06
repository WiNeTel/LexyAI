"""
PDF text extractor — uses pypdf, the lightweight pure-Python option.

We ignore image-only ("scanned") PDFs: pypdf returns empty strings for
those pages, which is fine — the upload still gets indexed (the user
sees "0 chars" and can re-upload as image for OCR via the Vision pipe).
"""

from __future__ import annotations

import io
from typing import Any


def parse_pdf(data: bytes, *, max_pages: int = 200) -> str:
    """Extract text from a PDF byte string.

    ``max_pages`` is a guard against pathological "10000-page-PDF"
    uploads — the embedder + chunker would happily process them and OOM
    the box. 200 pages is enough for practical documents.
    """
    if not data:
        return ""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover — pypdf is in requirements.txt
        return ""

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except Exception:
        # pypdf raises a long zoo of exception classes (PdfReadError,
        # ValueError, OSError, …). We collapse them all — the caller
        # treats empty text as "couldn't parse, store the file anyway".
        return ""

    pages: list[str] = []
    page_count = min(len(reader.pages), max_pages)
    for i in range(page_count):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages)
