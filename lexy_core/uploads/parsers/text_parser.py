"""Plain-text / markdown parser. Encoding-tolerant."""

from __future__ import annotations


# Encodings we try in order until one decodes without error.
# ``utf-8-sig`` is first because it strips the BOM that Notepad on
# Windows emits — the plain ``utf-8`` codec leaves the BOM as a literal
# U+FEFF and would silently decode without raising. cp1252 / latin-1 cover
# legacy Windows-saved text that isn't valid UTF-8.
_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def parse_text(data: bytes) -> str:
    """Decode plain text bytes. Returns the text, never raises."""
    if not data:
        return ""
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    # Last resort — replace undecodable bytes so we still return *something*.
    return data.decode("utf-8", errors="replace")
