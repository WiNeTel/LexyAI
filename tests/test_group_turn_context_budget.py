"""Integration test: GroupTurnOrchestrator reacts to live context_size changes.

This is the one behaviour the existing ``test_group_turn_sequential.py``
deliberately doesn't cover: the ``context_size_fn`` callback. We verify
that the orchestrator queries it per turn, so a config change between
calls takes effect immediately without a rebuild.
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
)


class _CapturingLLM:
    """Records the messages each call receives so we can inspect trim effects."""

    def __init__(self) -> None:
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
        self.calls.append({"messages": messages})
        system = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        if "Turn-Orchestrator" in system:
            return "luna"
        return "ok."

    def last_system(self) -> str:
        # Skip the order-picker call by finding a "Du bist" prompt.
        for call in reversed(self.calls):
            sys = next(
                (
                    m["content"]
                    for m in call["messages"]
                    if m["role"] == "system"
                ),
                "",
            )
            if "Du bist" in sys:
                return sys
        return ""


def _heavy_card() -> CharacterCard:
    """A card large enough that small budgets force visible trimming."""
    return CharacterCard(
        id="luna",
        name="Luna",
        persona="Luna ist ein komplexer Charakter. " * 200,
        example_dialog="*Luna macht etwas.* " * 100,
        age_stage="child",
    )


@pytest.mark.asyncio
async def test_orchestrator_reads_context_size_fn_per_turn() -> None:
    """Changing the callback return value between rounds takes effect
    immediately — no orchestrator rebuild required."""
    llm = _CapturingLLM()
    current_ctx = {"size": 16384}
    orch = GroupTurnOrchestrator(
        llm_chat=llm,
        context_size_fn=lambda: current_ctx["size"],
        turn_selection="autonomous",
    )

    req = GroupTurnRequest(
        session_id="s1",
        history=[],
        characters=[_heavy_card()],
        user_message="Hi",
    )

    # Round 1 at 16K — lots of room, no LOW drops expected
    await orch.run_round(req)
    big_sys = llm.last_system()

    # Round 2 at 2K — forces aggressive trimming
    current_ctx["size"] = 2048
    llm.calls.clear()
    await orch.run_round(req)
    small_sys = llm.last_system()

    # The 2K run's system prompt must be strictly smaller.
    assert len(small_sys) < len(big_sys)
    # MUST sections survived even under pressure.
    assert "Du bist Luna" in small_sys
    assert "## Regeln" in small_sys
    # example_dialog (LOW) dropped at 2K.
    assert "## Beispiel-Dialog" not in small_sys


@pytest.mark.asyncio
async def test_orchestrator_default_context_size_is_16k() -> None:
    """Orchestrator without a context_size_fn defaults to a safe 16K
    — no crash even if the plugin forgets to wire the callback."""
    llm = _CapturingLLM()
    orch = GroupTurnOrchestrator(llm_chat=llm, turn_selection="round_robin")
    await orch.run_round(
        GroupTurnRequest(
            session_id="s1",
            history=[],
            characters=[CharacterCard(id="lexy", name="Lexy", persona="k")],
            user_message="Hi",
        )
    )
    # It ran without exception — that's the whole contract.
    assert len(llm.calls) >= 1
