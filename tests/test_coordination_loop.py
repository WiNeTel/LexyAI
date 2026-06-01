"""Tests for the CoordinationLoop (read→act→verify→consequence cycle).

The whole point of the kernel: a scheduler-driven loop where commenting is
NOT enough — only a referee-confirmed action lowers a need, otherwise the
consequence escalates. Stubs stand in for the narration + referee LLM.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
import pytest

from lexy_core.coordination import (
    Attribute,
    Blackboard,
    CoordinationLoop,
    Referee,
    Threshold,
    WorldState,
)


def _baby_world() -> WorldState:
    ws = WorldState()
    ws.add_attribute(
        "scene",
        "baby",
        Attribute(
            name="hunger",
            value=65.0,
            rate_per_tick=5.0,
            thresholds=[
                Threshold(at=70.0, need="feed_baby", urgency=1),
                Threshold(at=100.0, need="baby_sick", urgency=3),
            ],
        ),
    )
    return ws


def _referee_llm(satisfied: bool, magnitude: float = 0.9):
    """Stub LLM for the Referee that always rules ``satisfied``."""
    payload = (
        '{"satisfied": %s, "magnitude": %s}'
        % ("true" if satisfied else "false", magnitude)
    )

    async def _f(messages: list[dict[str, str]], **kwargs: Any) -> str:
        return payload

    return _f


async def _feeds(demand: Any) -> str:
    return "Sie nimmt das Baby auf den Arm und stillt es ausgiebig."


async def _only_comments(demand: Any) -> str:
    return "Oh, das Baby schreit."


@pytest.mark.asyncio
async def test_satisfied_demand_closes_and_lowers_value() -> None:
    ws = _baby_world()
    loop = CoordinationLoop(ws, Referee(), _referee_llm(True, 0.9))

    report = await loop.tick("scene", _feeds)   # 65 -> 70 raises feed_baby; satisfied

    assert "baby:feed_baby" in report.satisfied
    # magnitude 0.9 → relieve 90 of span 100 → 70 - 90 clamped to 0
    assert ws.get("scene", "baby", "hunger") == 0.0
    assert loop.open_demands("scene") == []


@pytest.mark.asyncio
async def test_partial_magnitude_relieves_partially() -> None:
    ws = _baby_world()
    loop = CoordinationLoop(ws, Referee(), _referee_llm(True, 0.3))

    await loop.tick("scene", _feeds)            # 65 -> 70, satisfied 0.3 → -30
    assert ws.get("scene", "baby", "hunger") == 40.0


@pytest.mark.asyncio
async def test_commenting_keeps_demand_open_and_escalates() -> None:
    ws = _baby_world()
    loop = CoordinationLoop(ws, Referee(), _referee_llm(False))

    raised_needs: list[str] = []
    for _ in range(8):                          # 65 -> 100 over the run
        report = await loop.tick("scene", _only_comments)
        raised_needs.extend(d.need for d in report.raised)

    # The ignored hunger first raised feed_baby, then escalated to baby_sick.
    assert "feed_baby" in raised_needs
    assert "baby_sick" in raised_needs
    open_needs = {d.need for d in loop.open_demands("scene")}
    assert {"feed_baby", "baby_sick"} <= open_needs
    assert ws.get("scene", "baby", "hunger") == 100.0


@pytest.mark.asyncio
async def test_blackboard_receives_demand_and_decision_posts() -> None:
    db = await aiosqlite.connect(":memory:")
    try:
        bb = Blackboard(db)
        await bb.init_tables()
        ws = _baby_world()
        loop = CoordinationLoop(ws, Referee(), _referee_llm(True), blackboard=bb)

        await loop.tick("scene", _feeds)

        posts = await bb.read("scene")
        kinds = {p["kind"] for p in posts}
        assert "demand" in kinds      # world raised the obligation
        assert "decision" in kinds    # referee ruled on it
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_no_demand_no_drama() -> None:
    # Hunger well below threshold; a tick raises nothing and drives nothing.
    ws = _baby_world()
    ws.set("scene", "baby", "hunger", 10.0)
    loop = CoordinationLoop(ws, Referee(), _referee_llm(True))

    report = await loop.tick("scene", _feeds)   # 10 -> 15, no threshold
    assert report.raised == []
    assert report.satisfied == []
    assert loop.open_demands("scene") == []
