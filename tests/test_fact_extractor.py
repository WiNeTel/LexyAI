"""Tests for the coordination FactExtractor (stubbed LLM)."""

from __future__ import annotations

from typing import Any

import pytest

from lexy_core.coordination import FactExtractor


def _llm(reply: str):
    async def _f(messages: list[dict[str, str]], **kwargs: Any) -> str:
        return reply

    return _f


def _raises():
    async def _f(messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise RuntimeError("boom")

    return _f


@pytest.mark.asyncio
async def test_extracts_json() -> None:
    reply = '{"baby": {"held_by": "Shani", "location": "Wiege"}}'
    out = await FactExtractor().extract("…text…", "instr", _llm(reply))
    assert out == {"baby": {"held_by": "Shani", "location": "Wiege"}}


@pytest.mark.asyncio
async def test_strips_markdown_fence() -> None:
    reply = '```json\n{"baby": {"location": "Sofa"}}\n```'
    out = await FactExtractor().extract("x", "i", _llm(reply))
    assert out == {"baby": {"location": "Sofa"}}


@pytest.mark.asyncio
async def test_empty_text_short_circuits() -> None:
    out = await FactExtractor().extract("   ", "i", _llm('{"x":1}'))
    assert out == {}


@pytest.mark.asyncio
async def test_garbage_returns_empty() -> None:
    out = await FactExtractor().extract("x", "i", _llm("kein json"))
    assert out == {}


@pytest.mark.asyncio
async def test_non_object_returns_empty() -> None:
    out = await FactExtractor().extract("x", "i", _llm("[1,2,3]"))
    assert out == {}


@pytest.mark.asyncio
async def test_llm_exception_returns_empty() -> None:
    out = await FactExtractor().extract("x", "i", _raises())
    assert out == {}
