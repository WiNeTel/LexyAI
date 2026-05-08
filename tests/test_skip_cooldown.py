"""
Phase 13.2 — pin the skip-cooldown logic on CharacterChatPlugin.

When a character returns an empty/pass turn, the plugin records a
1-round cooldown so the LLM-orchestrator's speaker selection avoids
that character on the next round. Without this, "*Yara schweigt*"
loops 5+ times because the LLM keeps picking Yara as a candidate even
though her last three turns produced no content.

These tests exercise ``_record_skip_cooldowns`` and
``_tick_skip_cooldowns`` in isolation — the plugin instance is built
manually with stubbed deps so we don't need to spin up the whole
LexyApp.
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.character_chat.character_chat_plugin import (
    CharacterChatPlugin,
)


class _FakeAPI:
    """Minimal API stub — the cooldown methods don't touch it,
    but BasePlugin's __init__ stores it."""
    pass


class _FakeManifest:
    """Minimal manifest stub — only needed for the BasePlugin."""
    name = "character_chat"
    version = "0.1.0"
    config_defaults: dict[str, Any] = {}
    routing: dict[str, Any] = {}


class _FakeTurn:
    """Mimics ``CharacterTurn`` — only the two attributes the cooldown
    code reads."""
    def __init__(self, character_id: str, skipped: bool) -> None:
        self.character_id = character_id
        self.skipped = skipped


@pytest.fixture
def plugin() -> CharacterChatPlugin:
    return CharacterChatPlugin(_FakeAPI(), _FakeManifest())


class TestRecordSkipCooldowns:
    def test_skipped_chars_recorded(self, plugin: CharacterChatPlugin) -> None:
        plugin._record_skip_cooldowns("sess1", [
            _FakeTurn("yara", skipped=True),
            _FakeTurn("mira", skipped=False),
            _FakeTurn("lena", skipped=True),
        ])
        assert plugin._skip_cooldowns["sess1"] == {
            "yara": 1, "lena": 1,
        }

    def test_no_skipped_chars_no_entry(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        plugin._record_skip_cooldowns("sess1", [
            _FakeTurn("yara", skipped=False),
        ])
        assert "sess1" not in plugin._skip_cooldowns

    def test_empty_turns_list_is_noop(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        plugin._record_skip_cooldowns("sess1", [])
        assert "sess1" not in plugin._skip_cooldowns

    def test_per_session_isolation(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        plugin._record_skip_cooldowns("sess_a", [
            _FakeTurn("yara", skipped=True),
        ])
        plugin._record_skip_cooldowns("sess_b", [
            _FakeTurn("mira", skipped=True),
        ])
        # Each session has its own cooldown table.
        assert plugin._skip_cooldowns["sess_a"] == {"yara": 1}
        assert plugin._skip_cooldowns["sess_b"] == {"mira": 1}


class TestTickSkipCooldowns:
    def test_tick_excludes_then_decrements(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        """Snapshot-then-decrement contract: chars in the table at
        tick-time are excluded for THIS round, regardless of remaining
        count. Mira drops out next round, Yara still on cooldown."""
        plugin._skip_cooldowns["sess1"] = {"yara": 2, "mira": 1}
        excluded = plugin._tick_skip_cooldowns("sess1")
        assert excluded == {"yara", "mira"}
        # After tick: yara still has 1 round left, mira popped.
        assert plugin._skip_cooldowns["sess1"] == {"yara": 1}

    def test_tick_drops_session_when_empty(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        """Yara is excluded for THIS round (snapshot), then her counter
        decrements to 0 and she's popped. Next call: empty session."""
        plugin._skip_cooldowns["sess1"] = {"yara": 1}
        excluded = plugin._tick_skip_cooldowns("sess1")
        assert excluded == {"yara"}  # excluded this round
        assert "sess1" not in plugin._skip_cooldowns  # but pruned for next
        # And on the next round, no exclusions.
        excluded2 = plugin._tick_skip_cooldowns("sess1")
        assert excluded2 == set()

    def test_tick_unknown_session_returns_empty(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        excluded = plugin._tick_skip_cooldowns("does-not-exist")
        assert excluded == set()

    def test_record_then_tick_full_cycle(
        self, plugin: CharacterChatPlugin,
    ) -> None:
        """Round 1: yara silent → cooldown.
           Round 2 (tick): yara excluded.
           Round 2 ends with yara silent again → cooldown set fresh.
           Round 3 (tick): yara still excluded (re-set after round 2).
        """
        # Round 1 — yara silent.
        plugin._record_skip_cooldowns("sess1", [
            _FakeTurn("yara", skipped=True),
        ])
        # Round 2 — tick before round.
        excluded_2 = plugin._tick_skip_cooldowns("sess1")
        assert "yara" in excluded_2
        # Round 2 ends — yara silent again, re-record.
        plugin._record_skip_cooldowns("sess1", [
            _FakeTurn("yara", skipped=True),
        ])
        # Round 3 tick.
        excluded_3 = plugin._tick_skip_cooldowns("sess1")
        # Cooldown was reset to 1 by round 2's record; tick → still
        # excluded round 3.
        assert "yara" in excluded_3
