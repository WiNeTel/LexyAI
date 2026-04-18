"""Smoke tests for the FTS5 query sanitizer."""

from __future__ import annotations

from lexy_core.memory.memory_manager import _sanitize_fts_query


def test_plain_query_is_quoted() -> None:
    assert _sanitize_fts_query("hallo lexy") == '"hallo" OR "lexy"'


def test_punctuation_is_stripped() -> None:
    # Commas / colons / exclamation marks would otherwise crash FTS5 MATCH
    assert _sanitize_fts_query("Hallo, Lexy! Wie geht's?") == (
        '"Hallo" OR "Lexy" OR "Wie" OR "geht"'
    )


def test_german_umlauts_kept() -> None:
    assert _sanitize_fts_query("Grüße aus München") == (
        '"Grüße" OR "aus" OR "München"'
    )


def test_short_tokens_dropped() -> None:
    # Single-character tokens are noise for BM25
    assert _sanitize_fts_query("a b cc ddd") == '"cc" OR "ddd"'


def test_empty_when_only_punctuation() -> None:
    assert _sanitize_fts_query("!!!,,,???") == ""
    assert _sanitize_fts_query("") == ""
    assert _sanitize_fts_query("   ") == ""


def test_embedded_quotes_are_escaped() -> None:
    result = _sanitize_fts_query('say "hello" now')
    # The embedded quote characters inside tokens must be doubled.
    assert result == '"say" OR "hello" OR "now"'
