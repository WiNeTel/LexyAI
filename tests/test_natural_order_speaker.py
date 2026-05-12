"""
Phase 13.3 — pin the deterministic natural-order speaker activator.

SillyTavern's ``activateNaturalOrder`` (group-chats.js) inspired this:
no LLM call, just three signals — name-mention in the latest input,
a ``talkativeness`` roll, and a fall-through to "stay silent for the
caller to fall back to the LLM picker". These tests use a fixed
``random.seed`` so the rolls are reproducible.
"""

from __future__ import annotations

import random
import time

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
    _last_history_user_text,
)


def _card(name: str, talkativeness: float = 0.5) -> CharacterCard:
    return CharacterCard(
        name=name,
        persona=f"{name} persona",
        talkativeness=talkativeness,
        created_at=time.time(),
        updated_at=time.time(),
    )


async def _llm_stub(*_, **__) -> str:  # pragma: no cover (unused here)
    return ""


def _orch() -> GroupTurnOrchestrator:
    return GroupTurnOrchestrator(llm_chat=_llm_stub)


class TestNaturalOrderActivator:
    def test_name_mention_takes_pole_position(self) -> None:
        """When a name appears in the user message, that char is
        first regardless of talkativeness."""
        sandra = _card("Sandra", talkativeness=0.0)
        lena = _card("Lena", talkativeness=0.0)
        mira = _card("Mira", talkativeness=0.0)
        req = GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[sandra, lena, mira],
            user_message="Mira, kannst du mal nach Kokosnüssen schauen?",
        )
        random.seed(42)
        order = _orch()._activate_natural_order(
            req=req, eligible=[sandra, lena, mira],
        )
        # Mira is mentioned → activates first regardless of talkativeness=0.
        assert order[0] == mira.id

    def test_high_talkativeness_activates(self) -> None:
        """A char with talkativeness=1.0 always rolls in."""
        chatty = _card("Chatty", talkativeness=1.0)
        silent = _card("Silent", talkativeness=0.0)
        req = GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[chatty, silent],
            user_message="Lasst uns reden.",
        )
        random.seed(42)
        order = _orch()._activate_natural_order(
            req=req, eligible=[chatty, silent],
        )
        assert chatty.id in order
        assert silent.id not in order  # talkativeness 0.0 < any roll

    def test_zero_eligibles_returns_empty(self) -> None:
        req = GroupTurnRequest(
            session_id="s1", history=[], characters=[],
            user_message="Hallo?",
        )
        order = _orch()._activate_natural_order(req=req, eligible=[])
        assert order == []

    def test_no_activations_returns_empty(self) -> None:
        """All chars at 0.0 and no name-mention → empty list (caller
        should fall back to LLM picker)."""
        a = _card("Alpha", talkativeness=0.0)
        b = _card("Beta", talkativeness=0.0)
        req = GroupTurnRequest(
            session_id="s1", history=[], characters=[a, b],
            user_message="Hallo?",
        )
        random.seed(42)
        order = _orch()._activate_natural_order(req=req, eligible=[a, b])
        assert order == []

    def test_name_match_uses_word_boundary(self) -> None:
        """'Lena' should NOT match inside 'Galena' or 'Magdalena'."""
        lena = _card("Lena", talkativeness=0.0)
        req = GroupTurnRequest(
            session_id="s1", history=[], characters=[lena],
            user_message="Magdalena hat das Buch.",  # 'Lena' substring
        )
        random.seed(42)
        order = _orch()._activate_natural_order(req=req, eligible=[lena])
        # Word-boundary regex prevents the false positive. With
        # talkativeness=0, the natural-order returns nothing.
        assert order == []

    def test_haystack_falls_back_to_history_when_no_user_msg(self) -> None:
        """Pulse rounds have ``user_message=''`` — the activator
        should look at the most recent history entry instead."""
        mira = _card("Mira", talkativeness=0.0)
        history = [
            {"role": "assistant", "name": "Lexy", "content": "alles ok?"},
            {"role": "user", "content": "Mira soll mal kucken."},
        ]
        req = GroupTurnRequest(
            session_id="s1", history=history, characters=[mira],
            user_message="",  # pulse round, no fresh user input
        )
        random.seed(42)
        order = _orch()._activate_natural_order(req=req, eligible=[mira])
        # 'Mira' is in the last history user message → activated.
        assert order == [mira.id]


class TestLastHistoryUserText:
    def test_empty_history(self) -> None:
        assert _last_history_user_text([]) == ""

    def test_picks_last_user_msg(self) -> None:
        h = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _last_history_user_text(h) == "second"

    def test_skips_non_user_roles(self) -> None:
        h = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "reply1"},
            {"role": "system", "content": "noise"},
        ]
        # Last user message is "old" because the rest aren't role=user.
        assert _last_history_user_text(h) == "old"

    def test_handles_empty_content(self) -> None:
        h = [{"role": "user", "content": ""}]
        assert _last_history_user_text(h) == ""


class TestTalkativenessClamp:
    def test_clamps_above_one(self) -> None:
        c = CharacterCard(name="X", talkativeness=2.5)
        assert c.talkativeness == 1.0

    def test_clamps_below_zero(self) -> None:
        c = CharacterCard(name="X", talkativeness=-0.3)
        assert c.talkativeness == 0.0

    def test_default_is_half(self) -> None:
        c = CharacterCard(name="X")
        assert c.talkativeness == 0.5

    def test_garbage_falls_back_to_half(self) -> None:
        c = CharacterCard(name="X", talkativeness="not a number")  # type: ignore[arg-type]
        assert c.talkativeness == 0.5
