"""
Phase 13.3 — pin the SillyTavern-inspired group-nudge sections in
the prompt builder.

Three new sections land in every char turn:
1. ``group_roster``  (user, pre-history)   "[Gruppenchat. Anwesend: …]"
2. ``group_nudge``   (user, post-history)  "[Schreibe als <X>, reagiere
                                            konkret, verteilt Aufgaben]"
3. ``impersonation_guard`` (system, last)  "Du bist nicht der User…"

These tests build a minimal request and inspect the resulting
``PromptSection`` list — no LLM call.
"""

from __future__ import annotations

import time

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
)


def _card(name: str) -> CharacterCard:
    return CharacterCard(
        name=name,
        persona=f"{name} ist {name}.",
        talkativeness=0.5,
        created_at=time.time(),
        updated_at=time.time(),
    )


async def _llm_stub(*_, **__) -> str:  # pragma: no cover (unused)
    return ""


def _orch() -> GroupTurnOrchestrator:
    return GroupTurnOrchestrator(llm_chat=_llm_stub)


def _build_sections(card: CharacterCard, **req_kwargs):
    """Helper — runs the section builder and returns the list keyed
    by section name for fast lookup."""
    chars = req_kwargs.pop("characters", [card])
    history = req_kwargs.pop("history", [])
    previous_turns = req_kwargs.pop("previous_turns", [])
    req = GroupTurnRequest(
        session_id="s1",
        history=history,
        characters=chars,
        **req_kwargs,
    )
    sections = _orch()._build_turn_sections(
        card=card,
        req=req,
        all_cards=chars,
        previous_turns=previous_turns,
        own_memories=[],
    )
    return {s.name: s for s in sections}


class TestGroupRoster:
    def test_roster_lists_all_characters(self) -> None:
        sandra, lena, mira = _card("Sandra"), _card("Lena"), _card("Mira")
        sections = _build_sections(
            sandra, characters=[sandra, lena, mira],
            user_message="Hallo zusammen.",
        )
        roster = sections.get("group_roster")
        assert roster is not None
        assert "Sandra" in roster.text
        assert "Lena" in roster.text
        assert "Mira" in roster.text
        # Format: "[Gruppenchat. Anwesend: ...]"
        assert roster.text.startswith("[Gruppenchat")
        assert roster.text.endswith(".]")
        assert roster.role == "user"

    def test_roster_skipped_when_alone(self) -> None:
        """No peers → empty roster line is silly. Code should still
        emit it for consistency (just the speaker themselves)."""
        sandra = _card("Sandra")
        sections = _build_sections(sandra, characters=[sandra])
        roster = sections.get("group_roster")
        assert roster is not None
        assert "Sandra" in roster.text


class TestGroupNudge:
    def test_nudge_names_speaker_and_last_speaker(self) -> None:
        """Mira's turn after Lena spoke → nudge calls out both."""
        from plugins.character_chat.group_turn import CharacterTurn

        sandra, lena, mira = _card("Sandra"), _card("Lena"), _card("Mira")
        prev = [
            CharacterTurn(
                character_id=lena.id, character_name="Lena",
                content="Lena: Was sollen wir tun?", skipped=False, order=0,
            ),
        ]
        sections = _build_sections(
            mira, characters=[sandra, lena, mira],
            previous_turns=prev,
            user_message="",
        )
        nudge = sections.get("group_nudge")
        assert nudge is not None
        assert "Mira" in nudge.text
        assert "Lena" in nudge.text  # last speaker referenced
        # Anti-monologue + task-delegation hint must be in there.
        assert (
            "konkret" in nudge.text.lower()
            or "andere aufgabe" in nudge.text.lower()
        )
        # Format: bracketed system-style instruction.
        assert nudge.text.startswith("[")
        assert nudge.text.endswith("]")

    def test_nudge_first_speaker_no_predecessor(self) -> None:
        sandra = _card("Sandra")
        sections = _build_sections(sandra)
        nudge = sections.get("group_nudge")
        assert nudge is not None
        # No prev speaker → generic "react to last thing that happened".
        assert "Sandra" in nudge.text
        assert "[" in nudge.text and "]" in nudge.text

    def test_nudge_task_delegation_hint_present(self) -> None:
        """Mike's explicit ask: 'einer sammelt Holz, ein anderer
        Wasser'. Check the German distribution wording is in the
        nudge."""
        sandra = _card("Sandra")
        sections = _build_sections(sandra)
        nudge = sections.get("group_nudge")
        assert nudge is not None
        text_lower = nudge.text.lower()
        # at least one of the example tasks should appear, plus the
        # "nicht alle das gleiche"-style nudge.
        assert any(
            kw in text_lower
            for kw in ("holz", "wasser", "essen", "aufgabe")
        )
        assert "nicht alle" in text_lower or "andere" in text_lower


class TestAntiHallucination:
    """Phase 13.5 (C) — peer states + anti-hallucination wording.

    Mike's Castaway log: Sandra wrote "Mira tritt aus dem Schatten
    der Palmen mit einem Plastikbecher" while Mira was actually
    wading in the lagoon. The LLM had no way to know Mira's actual
    location. Two fixes are pinned here:
      1. The "## Andere Anwesende" block now embeds each peer's
         live ``location`` / ``last_action`` / ``mood``.
      2. The post-history ``group_nudge`` carries an explicit
         "Erfinde KEINE Aktionen für andere Charaktere" sentence.
    """

    def test_others_block_includes_peer_locations(self) -> None:
        sandra, mira = _card("Sandra"), _card("Mira")
        sections = _build_sections(
            sandra, characters=[sandra, mira],
            live_state_by_char={
                mira.id: {
                    "location": "lagune",
                    "last_action": "sucht_im_wasser_nach_treibgut",
                    "mood": "fokussiert",
                },
            },
        )
        others = sections.get("others")
        assert others is not None
        assert "Mira" in others.text
        # Peer's location, action, and mood must all surface.
        assert "lagune" in others.text
        assert "sucht_im_wasser_nach_treibgut" in others.text
        assert "fokussiert" in others.text
        # Anti-hallucination wording (the pin against story-driving).
        assert "Erfinde KEINE Aktionen" in others.text

    def test_others_block_falls_back_when_no_state(self) -> None:
        """Cards without state still render — just no 'currently' tail."""
        sandra, mira = _card("Sandra"), _card("Mira")
        sections = _build_sections(
            sandra, characters=[sandra, mira],
            live_state_by_char={},
        )
        others = sections.get("others")
        assert others is not None
        assert "Mira" in others.text
        # No phantom location string when state is missing.
        assert "Ort:" not in others.text

    def test_nudge_carries_anti_hallucination_sentence(self) -> None:
        sandra, mira = _card("Sandra"), _card("Mira")
        sections = _build_sections(sandra, characters=[sandra, mira])
        nudge = sections.get("group_nudge")
        assert nudge is not None
        # Explicit "don't invent peer actions" — Mike's #1 chat issue.
        assert "Erfinde KEINE Aktionen" in nudge.text


class TestImpersonationGuard:
    def test_guard_present_in_system(self) -> None:
        sandra = _card("Sandra")
        sections = _build_sections(sandra)
        guard = sections.get("impersonation_guard")
        assert guard is not None
        assert guard.role == "system"
        # Names the speaker explicitly.
        assert "Sandra" in guard.text
        # Mentions Mike (the user, by convention) so the LLM knows
        # who NOT to write as.
        assert "Mike" in guard.text or "User" in guard.text or "user" in guard.text


class TestSectionOrder:
    """Verify the new sections land in the right slots in the
    user/system block ordering."""

    def test_roster_before_history(self) -> None:
        sandra = _card("Sandra")
        sections = _build_sections(
            sandra, history=[
                {"role": "user", "content": "alt"},
            ],
        )
        order = list(sections.keys())
        # group_roster should appear before history (if both exist).
        if "history" in order and "group_roster" in order:
            assert order.index("group_roster") < order.index("history")

    def test_nudge_after_prev_turns_before_instruction(self) -> None:
        from plugins.character_chat.group_turn import CharacterTurn

        sandra, lena = _card("Sandra"), _card("Lena")
        prev = [
            CharacterTurn(
                character_id=lena.id, character_name="Lena",
                content="hi", skipped=False, order=0,
            ),
        ]
        sections = _build_sections(
            sandra, characters=[sandra, lena],
            previous_turns=prev,
        )
        order = list(sections.keys())
        # Pipeline: prev_turns → group_nudge → instruction.
        if "prev_turns" in order and "group_nudge" in order:
            assert order.index("prev_turns") < order.index("group_nudge")
        if "group_nudge" in order and "instruction" in order:
            assert order.index("group_nudge") < order.index("instruction")
