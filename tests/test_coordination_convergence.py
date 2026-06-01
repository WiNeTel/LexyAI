"""Tests for the coordination ConvergenceDetector (stubbed LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from lexy_core.coordination import ConvergenceDetector


def _make_llm(reply: str):
    """Return an async llm_chat stub that always answers ``reply``."""

    async def _llm(messages: list[dict[str, str]], **kwargs: Any) -> str:
        return reply

    return _llm


def _raises_llm():
    async def _llm(messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise RuntimeError("boom")

    return _llm


_CONTRIBS = [
    {"author": "analyst", "content": "Wir sollten Option A nehmen."},
    {"author": "critic", "content": "Stimme zu, Option A ist am robustesten."},
    {"author": "pragmatist", "content": "Option A, klar."},
]
_PARTICIPANTS = ["analyst", "critic", "pragmatist", "creative"]


@pytest.mark.asyncio
async def test_converges_when_agreements_meet_threshold() -> None:
    reply = (
        '{"agreements": ['
        '{"point": "Option A bevorzugt", "agreeing": ["analyst", "critic"]},'
        '{"point": "A ist robust", "agreeing": ["critic", "pragmatist"]}'
        "]}"
    )
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=2, llm_chat=_make_llm(reply))
    assert res.converged is True
    assert res.agreement_count == 2


@pytest.mark.asyncio
async def test_below_threshold_does_not_converge() -> None:
    reply = (
        '{"agreements": ['
        '{"point": "Option A bevorzugt", "agreeing": ["analyst", "critic"]}'
        "]}"
    )
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=3, llm_chat=_make_llm(reply))
    assert res.converged is False
    assert res.agreement_count == 1


@pytest.mark.asyncio
async def test_strips_markdown_fences() -> None:
    reply = (
        "```json\n"
        '{"agreements": [{"point": "p1", "agreeing": ["analyst", "critic"]},'
        '{"point": "p2", "agreeing": ["analyst", "pragmatist"]}]}'
        "\n```"
    )
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=2, llm_chat=_make_llm(reply))
    assert res.converged is True
    assert res.agreement_count == 2


@pytest.mark.asyncio
async def test_backwards_compat_agreeing_roles_key() -> None:
    reply = (
        '{"agreements": [{"point": "p", "agreeing_roles": ["analyst", "critic"]}]}'
    )
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=1, llm_chat=_make_llm(reply))
    assert res.agreement_count == 1


@pytest.mark.asyncio
async def test_single_agreeing_party_is_dropped() -> None:
    # Only one valid participant agrees → not a real agreement.
    reply = '{"agreements": [{"point": "p", "agreeing": ["analyst"]}]}'
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=1, llm_chat=_make_llm(reply))
    assert res.agreement_count == 0


@pytest.mark.asyncio
async def test_invalid_participants_filtered_out() -> None:
    # "ghost" is not in participants → only "analyst" remains → dropped.
    reply = '{"agreements": [{"point": "p", "agreeing": ["analyst", "ghost"]}]}'
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=1, llm_chat=_make_llm(reply))
    assert res.agreement_count == 0


@pytest.mark.asyncio
async def test_empty_contributions_short_circuit() -> None:
    det = ConvergenceDetector()
    res = await det.check([], _PARTICIPANTS, threshold=1, llm_chat=_make_llm("{}"))
    assert res.converged is False
    assert res.agreement_count == 0


@pytest.mark.asyncio
async def test_garbage_reply_yields_no_agreements() -> None:
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=1, llm_chat=_make_llm("kein json hier"))
    assert res.agreement_count == 0


@pytest.mark.asyncio
async def test_llm_exception_is_swallowed() -> None:
    det = ConvergenceDetector()
    res = await det.check(_CONTRIBS, _PARTICIPANTS, threshold=1, llm_chat=_raises_llm())
    assert res.converged is False
    assert res.agreement_count == 0
