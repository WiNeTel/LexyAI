"""Tests for RP reasoning capture + the display-only safety invariant.

When RP thinking is on, a character's chain-of-thought is captured and shown
collapsed in the chat bubble — but it must NEVER enter a prompt/history. These
tests pin the capture path AND prove the reasoning never leaks into a prompt.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lexy_core.llm.llm_client import LexyLLM
from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.character_chat_plugin import (
    _build_rp_history,
    _turn_to_public,
)
from plugins.character_chat.group_turn import (
    CharacterTurn,
    GroupTurnOrchestrator,
    GroupTurnRequest,
)


def _bob() -> CharacterCard:
    return CharacterCard(id="bob", name="Bob", persona="Nachbar.", age_stage="adult")


class _ContentLLM:
    """Content-only fake routed by a system-prompt substring."""

    def __init__(self, routes: dict[str, str]) -> None:
        self.routes = routes
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, messages, brain="e4b", max_tokens=200, temperature=0.5, **extra
    ) -> str:
        self.calls.append({"max_tokens": max_tokens, **extra})
        sys = " ".join(m["content"] for m in messages if m["role"] == "system")
        for key, val in self.routes.items():
            if key in sys:
                return val
        return ""


class _StructuredLLM:
    """`(content, reasoning)` fake for character turns."""

    def __init__(self, content: str, reasoning: str) -> None:
        self.content = content
        self.reasoning = reasoning
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, *, messages, brain="e4b", max_tokens=200, temperature=0.5, **extra
    ) -> tuple[str, str]:
        self.calls.append({"max_tokens": max_tokens, **extra})
        return self.content, self.reasoning


@pytest.mark.asyncio
async def test_chat_structured_splits_content_and_reasoning() -> None:
    llm = LexyLLM.__new__(LexyLLM)  # bypass __init__; only need the wrapper

    async def fake_stream(*, messages, brain="auto", **kw):
        for item in [
            ("reasoning", "I think "),
            ("reasoning", "therefore "),
            ("content", "Hi"),
            ("content", " there"),
        ]:
            yield item

    llm.chat_stream_structured = fake_stream  # type: ignore[assignment]
    content, reasoning = await llm.chat_structured(messages=[], brain="e4b")
    assert content == "Hi there"
    assert reasoning == "I think therefore "


@pytest.mark.asyncio
async def test_orchestrator_captures_reasoning_when_thinking_on() -> None:
    picker = _ContentLLM({"Turn-Orchestrator": "bob"})
    structured = _StructuredLLM(content="*nickt*", reasoning="(sie ist schon entkleidet)")
    orch = GroupTurnOrchestrator(
        llm_chat=picker,
        llm_chat_structured=structured,
        character_thinking=True,
    )
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1", history=[], characters=[_bob()], user_message="Hi",
        )
    )
    bob = next(t for t in result.turns if t.character_name == "Bob")
    assert bob.content == "*nickt*"
    assert bob.reasoning == "(sie ist schon entkleidet)"
    # The structured path was actually taken (thinking flows through).
    assert len(structured.calls) == 1
    assert structured.calls[0]["thinking"] is True


@pytest.mark.asyncio
async def test_thinking_off_uses_content_only_no_reasoning() -> None:
    picker = _ContentLLM({"Turn-Orchestrator": "bob", "Du bist Bob": "Hallo!"})
    structured = _StructuredLLM(content="X", reasoning="should-not-be-used")
    orch = GroupTurnOrchestrator(
        llm_chat=picker,
        llm_chat_structured=structured,
        character_thinking=False,
    )
    result = await orch.run_round(
        GroupTurnRequest(
            session_id="s1", history=[], characters=[_bob()], user_message="Hi",
        )
    )
    bob = next(t for t in result.turns if t.character_name == "Bob")
    assert bob.content == "Hallo!"
    assert bob.reasoning == ""
    assert structured.calls == []  # not used when thinking is off


def test_character_turn_reasoning_defaults_empty() -> None:
    t = CharacterTurn(character_id="c", character_name="C", content="hi")
    assert t.reasoning == ""


def test_turn_to_public_includes_reasoning() -> None:
    pub = _turn_to_public(
        CharacterTurn(character_id="c", character_name="C", content="hi", reasoning="rrr")
    )
    assert pub["reasoning"] == "rrr"
    assert pub["content"] == "hi"


def test_build_rp_history_excludes_reasoning() -> None:
    """SAFETY: reasoning on a row must NOT appear in the rebuilt prompt history."""
    rows = [
        SimpleNamespace(
            round_id="r1",
            trigger_kind="user",
            trigger_text="Hallo",
            character_name="Bob",
            content="Hi Mike",
            skipped=False,
            reasoning="SECRET-COT-must-not-leak",
        ),
    ]
    hist = _build_rp_history(rows)
    blob = repr(hist)
    assert "SECRET-COT-must-not-leak" not in blob
    assert any(d["content"] == "Hi Mike" for d in hist)
    # Each emitted entry is exactly {role, name, content} — no reasoning key.
    assert all(set(d.keys()) == {"role", "name", "content"} for d in hist)
