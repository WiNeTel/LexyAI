"""Tests for greeting handling (dedup + placeholder substitution)."""

from __future__ import annotations

from types import SimpleNamespace

from plugins.character_chat.character_chat_plugin import (
    _apply_placeholders,
    _has_greeting_turn,
)


def _row(char_id: str, trigger_kind: str = "user") -> SimpleNamespace:
    return SimpleNamespace(character_id=char_id, trigger_kind=trigger_kind)


def test_has_greeting_turn_detects_existing() -> None:
    rows = [_row("c1", "user"), _row("c1", "greeting")]
    assert _has_greeting_turn(rows, "c1") is True


def test_has_greeting_turn_absent() -> None:
    rows = [_row("c1", "user"), _row("c1", "pulse")]
    assert _has_greeting_turn(rows, "c1") is False
    assert _has_greeting_turn([], "c1") is False


def test_has_greeting_turn_is_per_character() -> None:
    rows = [_row("c2", "greeting")]
    assert _has_greeting_turn(rows, "c1") is False
    assert _has_greeting_turn(rows, "c2") is True


def test_apply_placeholders_basic() -> None:
    out = _apply_placeholders("Hi {{user}}, ich bin {{char}}.", "Shani")
    assert out == "Hi Mike, ich bin Shani."


def test_apply_placeholders_case_variants() -> None:
    assert _apply_placeholders("{{Char}} & {{User}}", "Shani") == "Shani & Mike"
    assert _apply_placeholders("{{CHAR}}/{{USER}}", "Shani") == "Shani/Mike"


def test_apply_placeholders_noop() -> None:
    assert _apply_placeholders("", "Shani") == ""
    assert _apply_placeholders("kein platzhalter", "Shani") == "kein platzhalter"
