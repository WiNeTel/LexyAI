"""Multi-format tool-call detection tests.

Covers every format the ToolCaller is expected to recognise. The registry is
pre-seeded with a dummy ``get_weather`` tool so unknown-tool filtering doesn't
get in the way.
"""

from __future__ import annotations

import pytest

from lexy_core.tools import ToolCaller, ToolRegistry


@pytest.fixture
def caller() -> ToolCaller:
    registry = ToolRegistry()
    registry.register(
        name="get_weather",
        handler=lambda location, units="metric": {"ok": True},
        schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"},
            },
            "required": ["location"],
        },
        description="Get weather",
        source="test",
    )
    registry.register(
        name="echo",
        handler=lambda text: {"echo": text},
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        description="Echo",
        source="test",
    )
    return ToolCaller(registry)


# ─── Format 1: Lexy native ───────────────────────────────────────────────────


def test_lexy_native_format(caller: ToolCaller) -> None:
    text = (
        "<tool_call>\n"
        '{"name": "get_weather", "arguments": {"location": "Hechthausen"}}\n'
        "</tool_call>"
    )
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"
    assert call.arguments == {"location": "Hechthausen"}


# ─── Format 2: ChatML / Qwen ─────────────────────────────────────────────────


def test_chatml_format(caller: ToolCaller) -> None:
    text = '<|tool_call|>{"name": "get_weather", "arguments": {"location": "Berlin"}}<|/tool_call|>'
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"
    assert call.arguments == {"location": "Berlin"}


# ─── Format 3: Gemma 4 native (this is what the user hit in production) ─────


def test_gemma_native_format_with_call_marker(caller: ToolCaller) -> None:
    # This is the actual output Lexy's 26B-A4B produced:
    #   <|tool_call>call\n{"name": "get_weather", "arguments": {"location": "Hechthausen"}}
    text = (
        "<|tool_call>call\n"
        '{"name": "get_weather", "arguments": {"location": "Hechthausen"}}\n'
    )
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"
    assert call.arguments == {"location": "Hechthausen"}


def test_gemma_native_format_without_marker(caller: ToolCaller) -> None:
    text = '<|tool_call|>{"name": "get_weather", "arguments": {"location": "Munich"}}'
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"


# ─── Format 4: ```tool_code fence ────────────────────────────────────────────


def test_tool_code_fence(caller: ToolCaller) -> None:
    text = (
        "Sure, let me check.\n"
        "```tool_code\n"
        '{"name": "get_weather", "arguments": {"location": "Paris"}}\n'
        "```"
    )
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"
    assert call.arguments == {"location": "Paris"}


# ─── Format 5: ```json fence with name/arguments shape ──────────────────────


def test_json_fence_with_shape(caller: ToolCaller) -> None:
    text = (
        "Here's the call:\n"
        "```json\n"
        '{"name": "echo", "arguments": {"text": "hello"}}\n'
        "```"
    )
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "echo"
    assert call.arguments == {"text": "hello"}


def test_json_fence_without_shape_ignored(caller: ToolCaller) -> None:
    # Regular code explanation with a JSON object → NOT a tool call
    text = (
        "Here's some example config:\n"
        "```json\n"
        '{"host": "localhost", "port": 8765}\n'
        "```"
    )
    assert caller.detect_tool_call(text) is None


# ─── Format 6: Bare JSON fallback ────────────────────────────────────────────


def test_bare_json_fallback(caller: ToolCaller) -> None:
    text = (
        'I will call the weather tool: {"name": "get_weather", '
        '"arguments": {"location": "Hamburg"}} and wait for the result.'
    )
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "get_weather"
    assert call.arguments == {"location": "Hamburg"}


def test_bare_json_with_nested_arguments(caller: ToolCaller) -> None:
    text = '{"name": "echo", "arguments": {"text": "a {brace} inside"}}'
    call = caller.detect_tool_call(text)
    assert call is not None
    assert call.name == "echo"
    assert call.arguments == {"text": "a {brace} inside"}


# ─── Unknown tool is dropped ────────────────────────────────────────────────


def test_unknown_tool_dropped(caller: ToolCaller) -> None:
    text = '<tool_call>{"name": "does_not_exist", "arguments": {}}</tool_call>'
    assert caller.detect_tool_call(text) is None


# ─── Multi-call detection ────────────────────────────────────────────────────


def test_detect_all_multiple_formats(caller: ToolCaller) -> None:
    text = (
        "First: <tool_call>"
        '{"name": "get_weather", "arguments": {"location": "Berlin"}}'
        "</tool_call>\n"
        "Second: ```tool_code\n"
        '{"name": "echo", "arguments": {"text": "hi"}}\n'
        "```"
    )
    calls = caller.detect_all(text)
    names = {call.name for call in calls}
    assert names == {"get_weather", "echo"}


# ─── strip_tool_call scrubs every wrapper ────────────────────────────────────


def test_strip_all_formats(caller: ToolCaller) -> None:
    text = (
        "Checking the weather...\n"
        "<tool_call>"
        '{"name": "get_weather", "arguments": {"location": "Berlin"}}'
        "</tool_call>\n"
        "<|tool_call|>"
        '{"name": "get_weather", "arguments": {"location": "Hamburg"}}'
        "<|/tool_call|>\n"
        "<|tool_call>call\n"
        '{"name": "get_weather", "arguments": {"location": "Munich"}}\n'
        "```tool_code\n"
        '{"name": "echo", "arguments": {"text": "hi"}}\n'
        "```"
    )
    stripped = caller.strip_tool_call(text)
    assert "<tool_call>" not in stripped
    assert "<|tool_call" not in stripped
    assert "tool_code" not in stripped
    assert '"get_weather"' not in stripped
    assert "Checking the weather" in stripped
