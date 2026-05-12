"""
Phase 13.7c — pin the sim-tick topic-picker.

The autonomous_sim path used to fire 1-speaker rounds with no user
message, which left the LLM nothing to react to and produced 'Was
ist los?' loops. The topic-picker now synthesises an anchor that
acts like a user-message: a recent real user message wins; absent
that, a critical tracked-stat (durst/hunger akut) drives the topic;
absent that, a random pick from a survival-themed backlog.

These tests pin the precedence chain at the unit level using a
minimal harness — a fake plugin object that exposes only what
``_pick_sim_topic`` reads.
"""

from __future__ import annotations

import asyncio
import time
import types
from typing import Any

import pytest

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.character_chat_plugin import (
    CharacterChatPlugin,
    _SIM_TOPIC_BACKLOG,
)


def _card(name: str, char_id: str | None = None) -> CharacterCard:
    return CharacterCard(
        id=char_id or name.lower(),
        name=name,
        persona=f"{name} ist {name}.",
        age_stage="adult",
        created_at=time.time(),
        updated_at=time.time(),
    )


class _FakeContainer:
    """Stub of RPSessionContainer with controllable get_char_state."""

    def __init__(self, states: dict[str, dict[str, str]]):
        self._states = states

    async def get_char_state(self, char_id: str) -> dict[str, str]:
        return self._states.get(char_id, {})


class _FakeStore:
    async def list_in_session(self, _session_id: str) -> list[CharacterCard]:
        return []


def _stub_plugin(
    *,
    history: list[dict[str, Any]] | None = None,
    container: _FakeContainer | None = None,
) -> CharacterChatPlugin:
    """Create an unbound plugin instance where only the bits the
    topic-picker reads are stubbed. Avoids the full plugin lifecycle
    (no asyncio loop, no DB, no ChromaDB)."""
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    # _load_session_history reads via api — short-circuit by binding
    # the method to return our fake list.
    plugin._load_session_history = lambda _sid: history or []
    plugin._store = _FakeStore()

    async def _get_container(_sid: str):
        return container

    plugin._get_rp_container = _get_container
    return plugin


# ─── Tests ─────────────────────────────────────────────────────────


class TestRecentUserMessage:
    def test_recent_user_message_wins(self) -> None:
        """User message < 30 min old wins over state and backlog."""
        now = time.time()
        history = [
            {"role": "assistant", "content": "Old assistant", "created_at": now - 100},
            {
                "role": "user",
                "content": "Wer übernimmt die erste Wache?",
                "created_at": now - 5 * 60,  # 5 min ago
            },
        ]
        plugin = _stub_plugin(history=history)
        topic = asyncio.run(
            plugin._pick_sim_topic("s1", [_card("Sandra")])
        )
        assert topic == "Wer übernimmt die erste Wache?"

    def test_old_user_message_falls_through(self) -> None:
        """User message > 30 min ago does not win — picker falls
        through to backlog (no critical state set)."""
        now = time.time()
        history = [
            {
                "role": "user",
                "content": "Was war das vorhin?",
                "created_at": now - 60 * 60,  # 60 min ago
            },
        ]
        plugin = _stub_plugin(history=history)
        topic = asyncio.run(
            plugin._pick_sim_topic("s1", [_card("Sandra")])
        )
        # Falls through to backlog; the original old message is NOT
        # returned.
        assert topic != "Was war das vorhin?"
        assert topic in _SIM_TOPIC_BACKLOG

    def test_no_history_falls_through(self) -> None:
        plugin = _stub_plugin(history=[])
        topic = asyncio.run(
            plugin._pick_sim_topic("s1", [_card("Sandra")])
        )
        assert topic in _SIM_TOPIC_BACKLOG


class TestStateDrivenTopic:
    def test_critical_thirst_triggers_state_topic(self) -> None:
        """Char with durst=akut → topic mentions her name and water."""
        sandra = _card("Sandra")
        container = _FakeContainer({sandra.id: {"durst": "akut", "hunger": "satt"}})
        plugin = _stub_plugin(history=[], container=container)
        topic = asyncio.run(plugin._pick_sim_topic("s1", [sandra]))
        assert "Sandra" in topic
        assert "Durst" in topic or "Wasser" in topic

    def test_critical_hunger_triggers_state_topic(self) -> None:
        lena = _card("Lena")
        container = _FakeContainer(
            {lena.id: {"durst": "neutral", "hunger": "akut"}}
        )
        plugin = _stub_plugin(history=[], container=container)
        topic = asyncio.run(plugin._pick_sim_topic("s1", [lena]))
        assert "Lena" in topic
        assert "Hunger" in topic or "Essen" in topic

    def test_first_critical_char_wins(self) -> None:
        """Multiple chars with critical stats — the first encountered
        wins. (Order = order of ``characters`` arg, which mirrors
        the attached-list order.)"""
        sandra = _card("Sandra")
        mira = _card("Mira")
        container = _FakeContainer({
            sandra.id: {"durst": "akut"},
            mira.id: {"durst": "akut"},
        })
        plugin = _stub_plugin(history=[], container=container)
        topic = asyncio.run(plugin._pick_sim_topic("s1", [sandra, mira]))
        assert "Sandra" in topic
        assert "Mira" not in topic

    def test_neutral_state_falls_through(self) -> None:
        sandra = _card("Sandra")
        container = _FakeContainer(
            {sandra.id: {"durst": "neutral", "hunger": "satt"}}
        )
        plugin = _stub_plugin(history=[], container=container)
        topic = asyncio.run(plugin._pick_sim_topic("s1", [sandra]))
        # No critical state → backlog fallback.
        assert topic in _SIM_TOPIC_BACKLOG


class TestBacklogFallback:
    def test_returns_one_of_the_backlog_items(self) -> None:
        plugin = _stub_plugin()
        topic = asyncio.run(plugin._pick_sim_topic("s1", []))
        assert topic in _SIM_TOPIC_BACKLOG
        assert len(topic) > 0

    def test_backlog_is_not_empty(self) -> None:
        """Defensive: someone deletes the constant → suite catches it."""
        assert len(_SIM_TOPIC_BACKLOG) >= 3
