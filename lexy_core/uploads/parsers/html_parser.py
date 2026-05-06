"""
HTML → plain-text parser.

Re-uses the regex-based stripper pattern from the web_crawler plugin so
we don't pull in BeautifulSoup just for upload parsing. The result is
not perfect (no DOM tree), but it's good enough to feed into the
knowledge chunker. If we ever need something better, this module is the
single place to swap the implementation.
"""

from __future__ import annotations

import html
import re


# Strip <script> and <style> blocks completely (with their content).
_RE_SCRIPT_STYLE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)
# Strip every other tag — we only want the visible text.
_RE_HTML_TAG = re.compile(r"<[^>]+>")
# Collapse runs of whitespace so the output isn't full of empty lines.
# We include U+00A0 (NBSP) here — `&nbsp;` decodes to that character, and
# leaving raw NBSP in the parsed text confuses downstream tokenizers.
_RE_WS = re.compile(r"[ \t\xa0]+")
_RE_BLANK_LINES = re.compile(r"\n\s*\n+")


def parse_html(data: bytes) -> str:
    """Strip HTML markup and return readable text.

    Decoding is permissive (UTF-8 with replacement) — most modern HTML
    is UTF-8, and silent garbage on a few characters is preferable to
    crashing on a single mis-encoded byte.
    """
    if not data:
        return ""
    raw = data.decode("utf-8", errors="replace")
    raw = _RE_SCRIPT_STYLE.sub(" ", raw)
    raw = _RE_HTML_TAG.sub(" ", raw)
    raw = html.unescape(raw)
    # Normalise whitespace.
    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = _RE_WS.sub(" ", line).strip()
        if cleaned:
            lines.append(cleaned)
    text = "\n".join(lines)
    return _RE_BLANK_LINES.sub("\n\n", text).strip()
