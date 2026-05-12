"""
Tests for the GroupTurnOrchestrator — the critical sequential-prompting
pattern from the character_chat design review.

We inject a deterministic fake LLM callable so we can assert *exactly*:

* Speaker N's prompt includes speakers 1..N-1's turns in the same round.
* ``[PASS]`` short-circuits a turn to ``skipped=True``.
* ``@Name`` mentions in the user message force that character to position 0.
* ``max_speakers_per_round`` actually clamps.
* ``round_robin`` mode is alphabetical and doesn't ask the LLM for an order.
* Turn budget is respected (one LLM call per non-orchestrator-pick turn).
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.group_turn import (
    CharacterTurn,
    GroupTurnOrchestrator,
    GroupTurnRequest,
)


# ─── Fake LLM ────────────────────────────────────────────────────────────────


class _FakeLLM:
    """Deterministic LLM stand-in.

    Records every call and returns scripted responses. Supports two flavours:

    * ``scripted``: a list of responses consumed in order.
    * ``route_by_system``: dict keyed by a substring of the system prompt;
      the first substring match picks the response. Useful when we don't
      know (or don't want to assume) the exact call order of order-picker
      vs speaker turns.
    """

    def __init__(
        self,
        *,
        scripted: list[str] | None = None,
        route_by_system: dict[str, str] | None = None,
    ) -> None:
        self.scripted = list(scripted or [])
        self.routes = dict(route_by_system or {})
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        messages: list[dict[str, str]],
        brain: str = "e4b",
        max_tokens: int = 200,
        temperature: float = 0.5,
        **_extras: Any,
    ) -> str:
        # ``_extras`` absorbs forward-compatible kwargs like ``thinking=False``
        # that the orchestrator passes through to the real LLMClient but we
        # don't care about recording here.
        self.calls.append(
            {
                "messages": messages,
                "brain": brain,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **_extras,
            }
        )
        # Try route first.
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        for key, reply in self.routes.items():
            if key in system:
                return reply
        if self.scripted:
            return self.scripted.pop(0)
        return ""


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _lexy() -> CharacterCard:
    return CharacterCard(
        id="lexy",
        name="Lexy",
        persona="KI-Assistentin, warmherzig, direkt.",
        age_stage="adult",
    )


def _luna(age: str = "baby") -> CharacterCard:
    return CharacterCard(
        id="luna",
        name="Luna",
        persona="Kleines Mädchen.",
        age_stage=age,
        relationships={"lexy": "Mutter"},
    )


def _bob() -> CharacterCard:
    return CharacterCard(
        id="bob",
        name="Bob",
        persona="Nachbar mit trockenem Humor.",
        age_stage="adult",
    )


# ─── Sequential prompting ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sequential_prompting_includes_previous_turns() -> None:
    """The critical test: turn N's prompt must contain turn N-1's content."""
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy, luna",  # order picker
            "Du bist Lexy": "Hallo Luna, alles gut?",
            "Du bist Luna": "Mama, ich hab Hunger.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _luna()],
        user_message="Hallo zusammen!",
    )
    result = await orch.run_round(req)
    assert result.speaker_order == ["lexy", "luna"]
    assert [t.character_name for t in result.turns] == ["Lexy", "Luna"]

    # Find the LLM call for Luna and check it contains Lexy's turn.
    luna_call = next(
        c for c in llm.calls
        if any("Du bist Luna" in m["content"] for m in c["messages"]
               if m["role"] == "system")
    )
    user_content = next(
        m["content"] for m in luna_call["messages"] if m["role"] == "user"
    )
    assert "Hallo Luna, alles gut?" in user_content
    assert "## Reaktionen dieser Runde" in user_content
    assert "Lexy" in user_content

    # Lexy's own prompt must NOT contain Luna's future reply.
    lexy_call = next(
        c for c in llm.calls
        if any("Du bist Lexy" in m["content"] for m in c["messages"]
               if m["role"] == "system")
    )
    lexy_user = next(
        m["content"] for m in lexy_call["messages"] if m["role"] == "user"
    )
    assert "Mama, ich hab Hunger" not in lexy_user


@pytest.mark.asyncio
async def test_pass_marker_marks_turn_skipped() -> None:
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "bob, lexy",
            "Du bist Bob": "[PASS]",
            "Du bist Lexy": "Du bist sehr still heute, Bob.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _bob()],
        user_message="Was meint ihr dazu?",
    )
    result = await orch.run_round(req)
    by_name = {t.character_name: t for t in result.turns}
    assert by_name["Bob"].skipped is True
    assert by_name["Bob"].content == ""
    assert by_name["Lexy"].skipped is False
    # Lexy's prompt should acknowledge Bob's silence via "*Bob schweigt*".
    lexy_call = next(
        c for c in llm.calls
        if any("Du bist Lexy" in m["content"] for m in c["messages"]
               if m["role"] == "system")
    )
    lexy_user = next(
        m["content"] for m in lexy_call["messages"] if m["role"] == "user"
    )
    assert "schweigt" in lexy_user


@pytest.mark.asyncio
async def test_at_mention_forces_speaker_to_front() -> None:
    """User writes '@Luna schau mal' → Luna speaks first, no matter what the LLM picks."""
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy, bob",  # picker tries to put Lexy first
            "Du bist Luna": "Ja, was denn?",
            "Du bist Lexy": "Luna, sei höflich.",
            "Du bist Bob": "Hehe.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _luna("child"), _bob()],
        user_message="@Luna schau mal her!",
    )
    result = await orch.run_round(req)
    assert result.speaker_order[0] == "luna"


@pytest.mark.asyncio
async def test_round_robin_mode_does_not_call_llm_for_order() -> None:
    """round_robin = deterministic alphabetical, no order-picker call."""
    llm = _FakeLLM(
        route_by_system={
            "Du bist Bob": "Hallo.",
            "Du bist Lexy": "Hi.",
            "Du bist Luna": "Hi!",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="round_robin")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _luna("child"), _bob()],
        user_message="Hi all.",
    )
    result = await orch.run_round(req)
    # Alphabetical by name: Bob, Lexy, Luna.
    assert [t.character_name for t in result.turns] == ["Bob", "Lexy", "Luna"]
    # No call had "Turn-Orchestrator" as system prompt.
    for call in llm.calls:
        systems = [
            m["content"] for m in call["messages"] if m["role"] == "system"
        ]
        assert not any("Turn-Orchestrator" in s for s in systems)


@pytest.mark.asyncio
async def test_max_speakers_per_round_clamps() -> None:
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy, luna, bob",
            "Du bist Lexy": "L",
            "Du bist Luna": "Lu",
            "Du bist Bob": "B",
        }
    )
    orch = GroupTurnOrchestrator(
        llm_chat=llm, turn_selection="autonomous", max_speakers_per_round=2
    )
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _luna("child"), _bob()],
        user_message="Wer will sprechen?",
    )
    result = await orch.run_round(req)
    assert len(result.turns) == 2
    assert result.speaker_order == ["lexy", "luna"]


@pytest.mark.asyncio
async def test_archived_characters_are_filtered_out() -> None:
    """Archived cards must not speak even if present in ``characters``."""
    luna = _luna("adult")
    luna.archived = True
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy",
            "Du bist Lexy": "Nur ich.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), luna],
        user_message="Hallo!",
    )
    result = await orch.run_round(req)
    assert [t.character_name for t in result.turns] == ["Lexy"]


@pytest.mark.asyncio
async def test_at_mention_caps_speaker_count_to_mention_count() -> None:
    """Phase 13.5 hotfix v3 — when the user @-mentions ONE char, that
    char alone speaks. The auto-fill that previously promoted a 2nd
    char into the slot via talkativeness/round-robin is suppressed
    when the user explicitly named someone.

    Mike's report: he writes '@Sandra hilf mir' and Lena ALSO
    answers because max_speakers_per_round=2. With identical
    tracked_state across both chars (e.g. arousal=extrem_notgeil),
    the parallel reply mirrors emotionally and feels like a copy.
    The cap stops the auto-fill at the named-count, restoring the
    explicit '/whisper-style' control the user expected.
    """
    llm = _FakeLLM(
        route_by_system={
            "Du bist Sandra": "Sandra: ich helfe dir.",
            "Du bist Lena": "Lena: ich auch.",
        }
    )
    sandra = CharacterCard(id="sandra", name="Sandra", talkativeness=0.5)
    lena = CharacterCard(id="lena", name="Lena", talkativeness=0.5)
    orch = GroupTurnOrchestrator(
        llm_chat=llm,
        max_speakers_per_round=2,  # default would let 2 in
        turn_selection="round_robin",
    )
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[sandra, lena],
        user_message="@Sandra hilf mir.",
    )
    result = await orch.run_round(req)
    # Only Sandra speaks — Lena does NOT auto-fill the second slot.
    assert [t.character_name for t in result.turns] == ["Sandra"]


@pytest.mark.asyncio
async def test_two_at_mentions_both_speak() -> None:
    """When the user names TWO chars, both speak. The cap follows
    the explicit count, not max_speakers."""
    llm = _FakeLLM(
        route_by_system={
            "Du bist Sandra": "Sandra: ich helfe.",
            "Du bist Lena": "Lena: ich auch.",
        }
    )
    sandra = CharacterCard(id="sandra", name="Sandra")
    lena = CharacterCard(id="lena", name="Lena")
    mira = CharacterCard(id="mira", name="Mira")
    orch = GroupTurnOrchestrator(
        llm_chat=llm,
        max_speakers_per_round=4,
        turn_selection="round_robin",
    )
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[sandra, lena, mira],
        user_message="@Sandra @Lena, was machen wir?",
    )
    result = await orch.run_round(req)
    # Both named chars speak; Mira (not named) stays silent.
    names = [t.character_name for t in result.turns]
    assert "Sandra" in names
    assert "Lena" in names
    assert "Mira" not in names


@pytest.mark.asyncio
async def test_no_mention_uses_full_max_speakers() -> None:
    """Without a user mention, the cap stays at max_speakers — the
    autonomous fill path is unchanged for non-targeted messages."""
    llm = _FakeLLM(
        route_by_system={
            "Du bist Sandra": "Sandra: ja.",
            "Du bist Lena": "Lena: ja.",
        }
    )
    sandra = CharacterCard(id="sandra", name="Sandra")
    lena = CharacterCard(id="lena", name="Lena")
    orch = GroupTurnOrchestrator(
        llm_chat=llm,
        max_speakers_per_round=2,
        turn_selection="round_robin",
    )
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[sandra, lena],
        user_message="Was sollen wir denn jetzt machen?",  # no @-name
    )
    result = await orch.run_round(req)
    # Both speak — no explicit mention → max_speakers cap applies.
    assert len(result.turns) == 2


@pytest.mark.asyncio
async def test_pulse_originator_appears_as_visible_turn() -> None:
    """Phase 13.5 (A): the pulse-from char now ALSO surfaces as the
    first turn, so the chat shows what they actually did (the pulse
    text). Without this fix, the pulse-trigger char (e.g. Yara firing
    every 10 min) never appears in the visible chat — only the
    reactions do — and the user can't tell what's happening.
    """
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy",
            "Du bist Lexy": "Luna, was ist denn?",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _luna("baby")],
        user_message="",
        pulse_from_id="luna",
        pulse_text="schreit laut",
    )
    result = await orch.run_round(req)
    # Luna appears FIRST with the pulse_text as her content; Lexy
    # follows as the regular LLM-generated reaction.
    names = [t.character_name for t in result.turns]
    assert names == ["Luna", "Lexy"]
    luna_turn = result.turns[0]
    assert luna_turn.content == "schreit laut"
    assert luna_turn.skipped is False
    assert luna_turn.character_id == "luna"
    # Lexy's prompt should still reference the pulse via the Impuls
    # section (kept for backwards compat — the LLM sees the pulse in
    # both places, but the duplicate is small and harmless).
    lexy_call = llm.calls[-1]
    lexy_user = next(
        m["content"] for m in lexy_call["messages"] if m["role"] == "user"
    )
    assert "Impuls" in lexy_user
    assert "schreit laut" in lexy_user


@pytest.mark.asyncio
async def test_pulse_originator_unknown_char_id_skips_visible_turn() -> None:
    """Defensive: if pulse_from_id points at a char not in
    ``req.characters`` (stale cache, race), don't crash — just skip
    the synthesised turn and proceed with the normal reactions."""
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy",
            "Du bist Lexy": "...",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_lexy(), _luna("baby")],
        user_message="",
        pulse_from_id="ghost-char-id-not-in-roster",
        pulse_text="something",
    )
    result = await orch.run_round(req)
    # No synthesised turn for the missing char; only Lexy's reaction.
    assert [t.character_name for t in result.turns] == ["Lexy"]


@pytest.mark.asyncio
async def test_empty_character_list_returns_empty_result() -> None:
    llm = _FakeLLM()
    orch = GroupTurnOrchestrator(llm_chat=llm)
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[],
            user_message="Hi",
        )
    )
    assert result.turns == []
    assert result.speaker_order == []
    assert llm.calls == []


@pytest.mark.asyncio
async def test_all_archived_returns_empty_result() -> None:
    luna = _luna("adult")
    luna.archived = True
    lexy = _lexy()
    lexy.archived = True
    orch = GroupTurnOrchestrator(llm_chat=_FakeLLM())
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[lexy, luna],
            user_message="Hi",
        )
    )
    assert result.turns == []


@pytest.mark.asyncio
async def test_llm_order_picker_falls_back_on_exception() -> None:
    """If the order-picker call raises, we default to round-robin over candidates."""
    calls_seen: list[int] = []

    async def flaky(*, messages: list[dict[str, str]], **kwargs: Any) -> str:
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        if "Turn-Orchestrator" in system:
            calls_seen.append(1)
            raise RuntimeError("LLM down")
        if "Du bist Lexy" in system:
            return "L"
        if "Du bist Luna" in system:
            return "Lu"
        return ""

    orch = GroupTurnOrchestrator(llm_chat=flaky, turn_selection="autonomous")
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[_lexy(), _luna("child")],
            user_message="Hallo",
        )
    )
    assert calls_seen == [1]  # we DID try to pick an order
    assert {t.character_name for t in result.turns} == {"Lexy", "Luna"}


@pytest.mark.asyncio
async def test_turn_level_llm_exception_marks_skipped() -> None:
    """If a single turn's LLM call fails, that character is skipped — not fatal."""

    async def selective(*, messages: list[dict[str, str]], **kwargs: Any) -> str:
        system = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        if "Turn-Orchestrator" in system:
            return "lexy, luna"
        if "Du bist Lexy" in system:
            raise RuntimeError("brain offline")
        if "Du bist Luna" in system:
            return "Ich sag trotzdem was."
        return ""

    orch = GroupTurnOrchestrator(llm_chat=selective, turn_selection="autonomous")
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[_lexy(), _luna("child")],
            user_message="Hallo",
        )
    )
    by_name = {t.character_name: t for t in result.turns}
    assert by_name["Lexy"].skipped is True
    assert by_name["Luna"].skipped is False
    assert by_name["Luna"].content == "Ich sag trotzdem was."


@pytest.mark.asyncio
async def test_history_tail_is_threaded_into_prompt() -> None:
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy",
            "Du bist Lexy": "Klar, erinnere ich mich.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    req = GroupTurnRequest(
        session_id="s1",
        history=[
            {"role": "user", "name": "Mike", "content": "Erinnerst du dich an den Schnee?"},
            {"role": "assistant", "name": "Lexy", "content": "Ja, der war magisch."},
        ],
        characters=[_lexy()],
        user_message="Und jetzt?",
    )
    await orch.run_round(req)
    lexy_call = next(
        c for c in llm.calls
        if any("Du bist Lexy" in m["content"] for m in c["messages"]
               if m["role"] == "system")
    )
    lexy_user = next(
        m["content"] for m in lexy_call["messages"] if m["role"] == "user"
    )
    assert "Bisheriger Chat" in lexy_user
    assert "Erinnerst du dich an den Schnee" in lexy_user


@pytest.mark.asyncio
async def test_turn_selection_invalid_raises() -> None:
    with pytest.raises(ValueError):
        GroupTurnOrchestrator(llm_chat=_FakeLLM(), turn_selection="cron")


@pytest.mark.asyncio
async def test_order_picker_can_return_names_not_ids() -> None:
    """Small models sometimes reply with names instead of ids. We normalise."""
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "Lexy, Luna",  # names, not ids
            "Du bist Lexy": "Hi.",
            "Du bist Luna": "Hi!",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[_lexy(), _luna("child")],
            user_message="Hallo",
        )
    )
    assert result.speaker_order == ["lexy", "luna"]


@pytest.mark.asyncio
async def test_order_picker_tolerates_numbered_list() -> None:
    """'1. luna\\n2. lexy' → ['luna', 'lexy']."""
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "1. luna\n2. lexy",
            "Du bist Luna": "Hi!",
            "Du bist Lexy": "Hi.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[_lexy(), _luna("child")],
            user_message="Hallo",
        )
    )
    assert result.speaker_order == ["luna", "lexy"]


@pytest.mark.asyncio
async def test_result_echoes_trigger_fields() -> None:
    llm = _FakeLLM(
        route_by_system={
            "Turn-Orchestrator": "lexy",
            "Du bist Lexy": "hm.",
        }
    )
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="autonomous")
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[_lexy()],
            user_message="test",
            pulse_text="",
        )
    )
    assert result.user_message == "test"
    assert result.pulse_from_id == ""
