"""Tests for the coordination Referee (game-master adjudication, stub LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from lexy_core.coordination import Demand, Referee


def _llm(reply: str):
    async def _f(messages: list[dict[str, str]], **kwargs: Any) -> str:
        return reply

    return _f


def _raising_llm():
    async def _f(messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise RuntimeError("boom")

    return _f


_DEMAND = Demand(
    scope="scene",
    entity="baby",
    attribute="hunger",
    need="feed_baby",
    urgency=1,
    value=72.0,
)


@pytest.mark.asyncio
async def test_concrete_action_is_satisfied() -> None:
    reply = '{"satisfied": true, "magnitude": 0.9, "rationale": "stillt das Baby"}'
    v = await Referee().adjudicate(_DEMAND, "Sie legt das Baby an und stillt es.", _llm(reply))
    assert v.satisfied is True
    assert v.magnitude == 0.9


@pytest.mark.asyncio
async def test_mere_comment_not_satisfied() -> None:
    reply = '{"satisfied": false, "magnitude": 0.0, "rationale": "nur kommentiert"}'
    v = await Referee().adjudicate(_DEMAND, "Oh, das Baby schreit.", _llm(reply))
    assert v.satisfied is False
    assert v.magnitude == 0.0


@pytest.mark.asyncio
async def test_empty_narration_short_circuits_to_unsatisfied() -> None:
    # No LLM call needed; fail-safe.
    v = await Referee().adjudicate(_DEMAND, "   ", _llm("{}"))
    assert v.satisfied is False


@pytest.mark.asyncio
async def test_markdown_fenced_json() -> None:
    reply = '```json\n{"satisfied": true, "magnitude": 0.5}\n```'
    v = await Referee().adjudicate(_DEMAND, "wechselt die Windel", _llm(reply))
    assert v.satisfied is True
    assert v.magnitude == 0.5


@pytest.mark.asyncio
async def test_magnitude_clamped() -> None:
    reply = '{"satisfied": true, "magnitude": 1.8}'
    v = await Referee().adjudicate(_DEMAND, "tut sehr viel", _llm(reply))
    assert v.magnitude == 1.0


@pytest.mark.asyncio
async def test_satisfied_with_zero_magnitude_gets_minimal_effect() -> None:
    # Contradictory verdict → treat as a full effect so the loop progresses.
    reply = '{"satisfied": true, "magnitude": 0.0}'
    v = await Referee().adjudicate(_DEMAND, "handelt", _llm(reply))
    assert v.satisfied is True
    assert v.magnitude == 1.0


@pytest.mark.asyncio
async def test_garbage_reply_fails_safe_to_unsatisfied() -> None:
    v = await Referee().adjudicate(_DEMAND, "irgendwas", _llm("kein json"))
    assert v.satisfied is False
    assert v.magnitude == 0.0


@pytest.mark.asyncio
async def test_llm_exception_fails_safe_to_unsatisfied() -> None:
    v = await Referee().adjudicate(_DEMAND, "irgendwas", _raising_llm())
    assert v.satisfied is False
    assert "llm_failed" in v.rationale


@pytest.mark.asyncio
async def test_non_numeric_magnitude_defaults_to_zero() -> None:
    reply = '{"satisfied": false, "magnitude": "viel"}'
    v = await Referee().adjudicate(_DEMAND, "x", _llm(reply))
    assert v.magnitude == 0.0
