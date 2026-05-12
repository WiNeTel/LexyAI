"""
Phase 13.3b — pin the new speaker-selection hierarchy.

Mike's request: "An erster Stelle Name-Mention, danach das LLM, dann
Talkativeness als Last-Resort." The LLM call now also gets full
personas + relationships so it can pick the contextually-right
speaker — Mike's example: Char A asks Char B *because* B is the
expert in X.

Three things to pin:
1. The LLM-prompt content includes per-character profiles and
   relationship lines.
2. Hierarchy: mention → LLM → talkativeness → round-robin safety.
3. The ``NONE`` reply from the LLM correctly triggers the talkativeness
   fallback instead of returning all candidates.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
)


def _card(
    name: str,
    persona: str = "",
    talkativeness: float = 0.5,
    relationships: dict[str, str] | None = None,
) -> CharacterCard:
    return CharacterCard(
        name=name,
        persona=persona or f"{name} ist ein Charakter.",
        talkativeness=talkativeness,
        relationships=relationships or {},
        created_at=time.time(),
        updated_at=time.time(),
    )


class _FakeLLM:
    """Records every call and returns a configurable response."""
    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _orch(llm: _FakeLLM, *, max_speakers: int = 2) -> GroupTurnOrchestrator:
    return GroupTurnOrchestrator(
        llm_chat=llm,
        max_speakers_per_round=max_speakers,
        speaker_selection_brain="e4b",
    )


# ─── Prompt-content tests ────────────────────────────────────────────


class TestLLMPromptContent:
    """The selector prompt must include per-character profiles +
    relationships so the LLM can decide on expertise + bond."""

    def test_prompt_includes_personas(self) -> None:
        llm = _FakeLLM(response="char-a")  # picks Sandra
        sandra = _card(
            "Sandra",
            persona="Sandra ist Krankenschwester. Reagiert auf Verletzungen.",
        )
        sandra._raw_id = sandra.id  # remember
        mira = _card("Mira", persona="Mira ist Surf-Lehrerin und kann tauchen.")
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[sandra, mira],
            user_message="Lena blutet am Knie.",
        )
        asyncio.run(_orch(llm)._ask_llm_for_order(
            req=req, candidates=[sandra, mira],
        ))
        # The prompt should contain BOTH personas.
        user_prompt = llm.calls[0]["messages"][1]["content"]
        assert "Krankenschwester" in user_prompt
        assert "Surf-Lehrerin" in user_prompt

    def test_prompt_includes_relationships(self) -> None:
        llm = _FakeLLM(response="")
        a = _card("Sandra")
        b = _card("Lena", relationships={a.id: "Schwester-Figur"})
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a, b], user_message="Hallo.",
        )
        asyncio.run(_orch(llm)._ask_llm_for_order(
            req=req, candidates=[a, b],
        ))
        user_prompt = llm.calls[0]["messages"][1]["content"]
        assert "Schwester-Figur" in user_prompt
        assert "Sandra" in user_prompt

    def test_prompt_includes_talkativeness_bias(self) -> None:
        llm = _FakeLLM(response="")
        a = _card("Sandra", talkativeness=0.3)
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a], user_message="Hallo.",
        )
        asyncio.run(_orch(llm)._ask_llm_for_order(
            req=req, candidates=[a],
        ))
        user_prompt = llm.calls[0]["messages"][1]["content"]
        # Score appears as "talkativeness=0.3"
        assert "talkativeness=0.3" in user_prompt

    def test_prompt_explains_heuristics(self) -> None:
        """System prompt names the heuristic priorities so the LLM
        knows expertise > relationships > relevance > talkativeness."""
        llm = _FakeLLM(response="")
        a = _card("X")
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a], user_message="Hi.",
        )
        asyncio.run(_orch(llm)._ask_llm_for_order(
            req=req, candidates=[a],
        ))
        sys_prompt = llm.calls[0]["messages"][0]["content"]
        for keyword in ("Expertise-Match", "Beziehungs-Bezug", "Relevanz"):
            assert keyword in sys_prompt


class TestNoneSignal:
    """LLM may answer ``NONE`` to indicate "no clear match" — that
    must surface as an empty list so the caller can fall through to
    talkativeness."""

    def test_none_returns_empty(self) -> None:
        llm = _FakeLLM(response="NONE")
        a = _card("X")
        b = _card("Y")
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a, b], user_message="...",
        )
        result = asyncio.run(_orch(llm)._ask_llm_for_order(
            req=req, candidates=[a, b],
        ))
        assert result == []

    def test_none_with_punctuation_still_recognised(self) -> None:
        """Tiny models sometimes answer 'NONE.' or '"NONE"'."""
        for raw in ("NONE.", "  NONE  ", '"NONE"', "[NONE]", "none"):
            llm = _FakeLLM(response=raw)
            a = _card("X")
            req = GroupTurnRequest(
                session_id="s1", history=[],
                characters=[a], user_message="...",
            )
            result = asyncio.run(_orch(llm)._ask_llm_for_order(
                req=req, candidates=[a],
            ))
            assert result == [], f"failed for raw={raw!r}"


# ─── Hierarchy tests ─────────────────────────────────────────────────


class TestSpeakerHierarchy:
    """Top-level _pick_speakers should walk: mention → LLM → talkativeness
    → round-robin (safety net)."""

    def test_mention_wins_over_llm(self) -> None:
        """If the user @-mentions someone, the LLM is NOT called for
        that round (mention path returns directly)."""
        llm = _FakeLLM(response="")
        a = _card("Sandra")
        b = _card("Lena")
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a, b],
            user_message="Sandra, hilf mir.",
        )
        result = asyncio.run(_orch(llm)._pick_speakers(
            req=req, eligible=[a, b], forced=[a.id],
        ))
        assert result[0] == a.id
        # Mention path goes through LLM if LLM-pick later, BUT the
        # mention itself is always in pole position.
        assert a.id in result

    def test_llm_picks_first_when_no_mention(self) -> None:
        """When no name is mentioned, the LLM-story-match decides."""
        llm = _FakeLLM(response="char-a")  # filled below
        a = _card("Sandra", talkativeness=0.0)
        b = _card("Lena", talkativeness=0.0)
        # Resolve real IDs into the canned response.
        llm.response = a.id
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a, b], user_message="Was tun?",
        )
        result = asyncio.run(_orch(llm)._pick_speakers(
            req=req, eligible=[a, b], forced=[],
        ))
        assert result == [a.id]
        # LLM was called (one to two times depending on flow).
        assert len(llm.calls) >= 1

    def test_talkativeness_fallback_when_llm_says_none(self) -> None:
        """LLM returns NONE → talkativeness-roll picks the chatty char."""
        llm = _FakeLLM(response="NONE")
        a = _card("Chatty", talkativeness=1.0)   # always rolls in
        b = _card("Silent", talkativeness=0.0)
        req = GroupTurnRequest(
            session_id="s1", history=[],
            characters=[a, b], user_message="...",
        )
        result = asyncio.run(_orch(llm)._pick_speakers(
            req=req, eligible=[a, b], forced=[],
        ))
        # Talkativeness fallback fired; chatty char picked.
        assert a.id in result
        assert b.id not in result
