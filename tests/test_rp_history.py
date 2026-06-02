"""Tests for _build_rp_history — the RP transcript reconstruction.

This is the fix for characters "forgetting" recent facts: the round history
must include the characters' own + others' prior-round turns, not just the
user's lines.
"""

from __future__ import annotations

from types import SimpleNamespace

from plugins.character_chat.character_chat_plugin import _build_rp_history


def _row(round_id, character_name, content, *, trigger_kind="user",
         trigger_text="", skipped=False):
    return SimpleNamespace(
        round_id=round_id,
        character_name=character_name,
        content=content,
        trigger_kind=trigger_kind,
        trigger_text=trigger_text,
        skipped=skipped,
    )


def test_user_line_then_character_turn() -> None:
    rows = [_row("r1", "Shani", "Das ist Mia.", trigger_text="Wie heisst das Baby?")]
    assert _build_rp_history(rows) == [
        {"role": "user", "name": "Mike", "content": "Wie heisst das Baby?"},
        {"role": "assistant", "name": "Shani", "content": "Das ist Mia."},
    ]


def test_multiple_characters_and_rounds_in_order() -> None:
    rows = [
        _row("r1", "Shani", "A", trigger_text="hi"),
        _row("r1", "Greta", "B", trigger_text="hi"),
        _row("r2", "Shani", "C", trigger_text="ok"),
    ]
    h = _build_rp_history(rows)
    assert [e["content"] for e in h] == ["hi", "A", "B", "ok", "C"]
    assert [e["name"] for e in h] == ["Mike", "Shani", "Greta", "Mike", "Shani"]


def test_skipped_and_empty_turns_dropped() -> None:
    rows = [
        _row("r1", "Shani", "", trigger_text="hi"),            # empty
        _row("r1", "Greta", "B", trigger_text="hi", skipped=True),  # skipped
        _row("r1", "Mira", "C", trigger_text="hi"),
    ]
    assert [e["content"] for e in _build_rp_history(rows)] == ["hi", "C"]


def test_pulse_round_has_no_user_line() -> None:
    rows = [_row("r1", "Shani", "reagiert", trigger_kind="pulse",
                 trigger_text="[baby] schreit")]
    assert _build_rp_history(rows) == [
        {"role": "assistant", "name": "Shani", "content": "reagiert"}
    ]


def test_round_limit_keeps_most_recent() -> None:
    rows = [_row(f"r{i}", "Shani", f"t{i}", trigger_text=f"u{i}") for i in range(10)]
    h = _build_rp_history(rows, limit_rounds=2)
    assert [e["content"] for e in h] == ["u8", "t8", "u9", "t9"]


def test_per_turn_char_cap() -> None:
    rows = [_row("r1", "Shani", "x" * 1000, trigger_text="hi")]
    h = _build_rp_history(rows, max_chars_per_turn=10)
    assert len(h[1]["content"]) == 10


def test_empty_rows() -> None:
    assert _build_rp_history([]) == []
