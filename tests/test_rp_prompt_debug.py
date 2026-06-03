"""Tests for the opt-in prompt-debug logging in group_turn.

The feature prints the exact system/user prompt (and raw response) sent to the
LLM when ``LEXY_DEBUG_PROMPTS`` is truthy. Default OFF so normal runs stay
quiet. These tests pin the env-flag parsing and the no-op/emit behaviour.
"""

from __future__ import annotations

import logging

import pytest

from plugins.character_chat.group_turn import (
    _emit_prompt_debug,
    _emit_response_debug,
    _prompt_debug_enabled,
)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On", " on "])
def test_prompt_debug_enabled_truthy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LEXY_DEBUG_PROMPTS", value)
    assert _prompt_debug_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "maybe"])
def test_prompt_debug_enabled_falsy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LEXY_DEBUG_PROMPTS", value)
    assert _prompt_debug_enabled() is False


def test_prompt_debug_enabled_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEXY_DEBUG_PROMPTS", raising=False)
    assert _prompt_debug_enabled() is False


def test_emit_prompt_debug_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("LEXY_DEBUG_PROMPTS", raising=False)
    with caplog.at_level(logging.INFO, logger="plugins.character_chat.group_turn"):
        _emit_prompt_debug(
            character="Shani",
            brain="e4b",
            system_prompt="SYS",
            user_content="USR",
            max_tokens=512,
            temperature=0.8,
        )
    assert caplog.records == []


def test_emit_prompt_debug_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LEXY_DEBUG_PROMPTS", "1")
    with caplog.at_level(logging.INFO, logger="plugins.character_chat.group_turn"):
        _emit_prompt_debug(
            character="Shani",
            brain="e4b",
            system_prompt="MY-SYSTEM-PROMPT",
            user_content="MY-USER-CONTENT",
            max_tokens=512,
            temperature=0.8,
        )
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "PROMPT DEBUG" in blob
    assert "character=Shani" in blob
    assert "MY-SYSTEM-PROMPT" in blob
    assert "MY-USER-CONTENT" in blob


def test_emit_response_debug_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LEXY_DEBUG_PROMPTS", "yes")
    with caplog.at_level(logging.INFO, logger="plugins.character_chat.group_turn"):
        _emit_response_debug(character="Shani", content="HELLO-WORLD", skipped=False)
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "LLM RESPONSE" in blob
    assert "HELLO-WORLD" in blob


def test_emit_response_debug_empty_shows_placeholder(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LEXY_DEBUG_PROMPTS", "1")
    with caplog.at_level(logging.INFO, logger="plugins.character_chat.group_turn"):
        _emit_response_debug(character="Shani", content="", skipped=True)
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "(empty)" in blob
    assert "skipped=True" in blob
