"""Tests for the coordination SceneDirector (stubbed LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from lexy_core.coordination import SceneDirector, looks_like_has_dependent


def _llm(reply: str):
    async def _f(messages: list[dict[str, str]], **kwargs: Any) -> str:
        return reply

    return _f


def test_keyword_prefilter() -> None:
    assert looks_like_has_dependent("Sie hat ein 2 Monate altes Baby")
    assert looks_like_has_dependent("ist schwanger im 8. Monat")
    assert not looks_like_has_dependent("Ein einsamer Söldner ohne Bindungen")


@pytest.mark.asyncio
async def test_analyze_returns_sanitised_needs() -> None:
    reply = (
        '{"needs": [{"entity": "baby", "attribute": "hunger", '
        '"rate_per_minute": 2, "caregiver": "Shani", '
        '"thresholds": [{"at": 70, "need": "feed_baby", "urgency": 1}]}], '
        '"note": "Baby erkannt"}'
    )
    out = await SceneDirector().analyze(
        persona="Shani hat ein Baby", llm_chat=_llm(reply)
    )
    assert len(out["needs"]) == 1
    n = out["needs"][0]
    assert n["entity"] == "baby" and n["attribute"] == "hunger"
    assert n["caregiver"] == "Shani"
    assert n["thresholds"][0]["need"] == "feed_baby"
    assert out["note"] == "Baby erkannt"


@pytest.mark.asyncio
async def test_analyze_drops_malformed_needs() -> None:
    reply = (
        '{"needs": ['
        '{"attribute": "hunger", "thresholds": [{"at": 70, "need": "x"}]},'  # no entity
        '{"entity": "baby", "attribute": "hunger"},'                         # no thresholds
        '{"entity": "baby", "attribute": "sleep", '
        '"thresholds": [{"at": 80, "need": "nap"}]}'                         # valid
        ']}'
    )
    out = await SceneDirector().analyze(persona="x", llm_chat=_llm(reply))
    assert [n["attribute"] for n in out["needs"]] == ["sleep"]


@pytest.mark.asyncio
async def test_analyze_empty_on_no_needs() -> None:
    out = await SceneDirector().analyze(persona="x", llm_chat=_llm('{"needs": []}'))
    assert out["needs"] == []


@pytest.mark.asyncio
async def test_analyze_empty_on_garbage() -> None:
    out = await SceneDirector().analyze(persona="x", llm_chat=_llm("kein json"))
    assert out["needs"] == []
