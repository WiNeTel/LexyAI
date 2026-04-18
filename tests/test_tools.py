"""Smoke tests for the tool system + dirty JSON parser."""

from __future__ import annotations

import pytest

from lexy_core.llm import parse_dirty_json
from lexy_core.tools import ToolCaller, ToolRegistry


@pytest.mark.asyncio
async def test_tool_registry_execute() -> None:
    registry = ToolRegistry()

    async def echo(text: str) -> dict[str, str]:
        return {"echo": text}

    registry.register(
        name="echo",
        handler=echo,
        schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        description="Echo a string",
        source="test",
    )
    assert registry.has_tools()
    result = await registry.execute("echo", {"text": "hi"})
    assert result.success
    assert result.data == {"echo": "hi"}


@pytest.mark.asyncio
async def test_tool_caller_detect_and_strip() -> None:
    registry = ToolRegistry()
    registry.register(
        name="echo",
        handler=lambda text: {"echo": text},
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        description="",
        source="test",
    )
    caller = ToolCaller(registry)

    llm_output = (
        "Sure, let me use the tool.\n"
        "<tool_call>\n"
        '{"name": "echo", "arguments": {"text": "hi"}}\n'
        "</tool_call>\n"
    )
    call = caller.detect_tool_call(llm_output)
    assert call is not None
    assert call.name == "echo"
    assert call.arguments == {"text": "hi"}

    stripped = caller.strip_tool_call(llm_output)
    assert "<tool_call>" not in stripped
    assert "let me use the tool" in stripped


def test_dirty_json_fallback() -> None:
    # Valid JSON
    assert parse_dirty_json('{"a": 1}') == {"a": 1}
    # Python literals
    assert parse_dirty_json('{"a": True, "b": None}') == {"a": True, "b": None}
    # Trailing comma
    assert parse_dirty_json('{"a": 1,}') == {"a": 1}
    # Code fence
    assert parse_dirty_json('```json\n{"a": 1}\n```') == {"a": 1}
    # Garbage
    assert parse_dirty_json("not json at all") is None
