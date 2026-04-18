"""
Tests for LexyLLM.chat_stream_structured — Gemma 4 reasoning parsing.

We plug an httpx MockTransport into the internal AsyncClient so every
request is answered with a pre-scripted SSE stream. No real server, no
LiteLLM, no network.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

import httpx
import pytest

from lexy_core.config import BrainConfig, LexyConfig, RoutingConfig
from lexy_core.llm.llm_client import LexyLLM


# ── SSE helpers ─────────────────────────────────────────────────────────────


def _sse_chunk(**delta_fields: Any) -> bytes:
    """Build one SSE 'data: {...}\\n\\n' line that wraps a delta dict."""
    payload = {"choices": [{"delta": delta_fields, "index": 0}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_body(pieces: Iterable[bytes]) -> bytes:
    return b"".join(pieces) + b"data: [DONE]\n\n"


def _make_mock_transport(body: bytes) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


def _build_llm_with_body(body: bytes) -> LexyLLM:
    brain = BrainConfig(model="gemma-4-e4b-it", endpoint="http://127.0.0.1:5006/v1")
    config = LexyConfig(
        brains={"e4b": brain},
        routing=RoutingConfig(default_brain="e4b"),
    )
    llm = LexyLLM(config)
    # Pre-populate the per-brain client with a mock-transport backed one so
    # ``connect()`` is unnecessary for tests.
    llm._clients["e4b"] = httpx.AsyncClient(
        base_url=brain.endpoint.rstrip("/"),
        transport=_make_mock_transport(body),
    )
    return llm


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reasoning_content_field_routed_separately() -> None:
    """Explicit ``reasoning_content`` delta lands on the reasoning channel."""
    body = _sse_body(
        [
            _sse_chunk(role="assistant"),
            _sse_chunk(reasoning_content="Step 1: understand. "),
            _sse_chunk(reasoning_content="Step 2: answer."),
            _sse_chunk(content="Die Antwort lautet 42."),
        ]
    )
    llm = _build_llm_with_body(body)

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    async for kind, text in llm.chat_stream_structured(
        messages=[{"role": "user", "content": "q?"}],
        brain="e4b",
    ):
        if kind == "reasoning":
            reasoning_parts.append(text)
        else:
            content_parts.append(text)

    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    assert "Step 1" in reasoning
    assert "Step 2" in reasoning
    assert content == "Die Antwort lautet 42."


@pytest.mark.asyncio
async def test_gemma_channel_thought_tag_is_split() -> None:
    """
    Gemma 4 native reasoning block: ``<|channel>thought ... <channel|>``.
    The reasoning text must land on the reasoning channel and disappear
    from the content stream.
    """
    body = _sse_body(
        [
            _sse_chunk(content="<|channel>thought\n"),
            _sse_chunk(content="Der Nutzer fragt nach dem Wetter. "),
            _sse_chunk(content="Ich rufe das Tool auf.\n<channel|>"),
            _sse_chunk(content="Das Wetter ist sonnig."),
        ]
    )
    llm = _build_llm_with_body(body)

    reasoning = ""
    content = ""
    async for kind, text in llm.chat_stream_structured(
        messages=[{"role": "user", "content": "wetter?"}],
        brain="e4b",
    ):
        if kind == "reasoning":
            reasoning += text
        else:
            content += text

    assert "Der Nutzer fragt" in reasoning
    assert "Tool auf" in reasoning
    assert "<|channel>" not in content
    assert "<channel|>" not in content
    assert "Das Wetter ist sonnig." in content


@pytest.mark.asyncio
async def test_gemma_tag_split_across_chunks() -> None:
    """The opener ``<|channel>thought`` must survive chunk boundaries."""
    body = _sse_body(
        [
            _sse_chunk(content="Hi <|chan"),
            _sse_chunk(content="nel>thought\nreasoning here"),
            _sse_chunk(content="<channel|>Hallo!"),
        ]
    )
    llm = _build_llm_with_body(body)

    reasoning = ""
    content = ""
    async for kind, text in llm.chat_stream_structured(
        messages=[{"role": "user", "content": "x"}],
        brain="e4b",
    ):
        if kind == "reasoning":
            reasoning += text
        else:
            content += text

    assert "reasoning here" in reasoning
    assert "<|channel>" not in content
    assert "Hi " in content
    assert "Hallo!" in content


@pytest.mark.asyncio
async def test_legacy_think_tag_still_works() -> None:
    """Legacy Qwen/DeepSeek ``<think>...</think>`` tags are still recognised."""
    body = _sse_body(
        [
            _sse_chunk(content="<think>legacy reasoning</think>"),
            _sse_chunk(content="Final answer here."),
        ]
    )
    llm = _build_llm_with_body(body)

    reasoning = ""
    content = ""
    async for kind, text in llm.chat_stream_structured(
        messages=[{"role": "user", "content": "x"}],
        brain="e4b",
    ):
        if kind == "reasoning":
            reasoning += text
        else:
            content += text

    assert "legacy reasoning" in reasoning
    assert "<think>" not in content
    assert "Final answer here." in content


@pytest.mark.asyncio
async def test_no_tags_is_plain_content() -> None:
    """Content without any reasoning markers flows through verbatim."""
    body = _sse_body(
        [
            _sse_chunk(content="Das"),
            _sse_chunk(content=" ist"),
            _sse_chunk(content=" ein"),
            _sse_chunk(content=" Test."),
        ]
    )
    llm = _build_llm_with_body(body)

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    async for kind, text in llm.chat_stream_structured(
        messages=[{"role": "user", "content": "x"}],
        brain="e4b",
    ):
        if kind == "reasoning":
            reasoning_parts.append(text)
        else:
            content_parts.append(text)

    assert reasoning_parts == []
    assert "".join(content_parts) == "Das ist ein Test."


# ── System-prompt injection for thinking mode ──────────────────────────────


def test_inject_thinking_token_creates_system_message() -> None:
    messages = [{"role": "user", "content": "hi"}]
    out = LexyLLM._inject_thinking_token(messages)
    assert out[0]["role"] == "system"
    assert "<|think|>" in out[0]["content"]
    assert out[1]["role"] == "user"


def test_inject_thinking_token_prepends_to_existing_system() -> None:
    messages = [
        {"role": "system", "content": "Du bist Lexy."},
        {"role": "user", "content": "hi"},
    ]
    out = LexyLLM._inject_thinking_token(messages)
    assert out[0]["content"].startswith("<|think|>")
    assert "Du bist Lexy." in out[0]["content"]
    assert len(out) == 2


def test_inject_thinking_token_idempotent() -> None:
    messages = [
        {"role": "system", "content": "<|think|>\nDu bist Lexy."},
        {"role": "user", "content": "hi"},
    ]
    out = LexyLLM._inject_thinking_token(messages)
    # Should not double-prefix
    assert out[0]["content"].count("<|think|>") == 1
