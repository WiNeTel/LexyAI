"""
Tests for LexyAgent.process_stream (streaming tool loop).

Uses a lightweight fake LexyApp that injects a scripted stream of LLM
responses per iteration so we can exercise every branch of the look-ahead
state machine without talking to a real model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import pytest

from lexy_core.agent import SessionStore
from lexy_core.agent.agent import LexyAgent
from lexy_core.events import EventBus, HookManager, LexySignals
from lexy_core.memory import MemoryManager
from lexy_core.tools import ToolCaller, ToolRegistry


# ── Minimal fakes ───────────────────────────────────────────────────────────


class FakeLLM:
    """
    Scripted LLM fake. ``responses`` is a list of full-text replies, one per
    LLM call (chat_stream splits them into char-sized chunks).

    ``seen_messages`` captures every ``messages`` list passed to chat/stream
    so tests can assert that history was injected correctly.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._calls = 0
        self.seen_messages: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict], brain: str = "auto", **kw: Any) -> str:
        self.seen_messages.append([dict(m) for m in messages])
        text = self._pop()
        return text

    async def chat_stream(
        self, messages: list[dict], brain: str = "auto", **kw: Any
    ) -> AsyncIterator[str]:
        self.seen_messages.append([dict(m) for m in messages])
        text = self._pop()
        # Stream 12-char chunks so the look-ahead window has something to see
        for idx in range(0, len(text), 12):
            yield text[idx : idx + 12]

    async def chat_stream_structured(
        self, messages: list[dict], brain: str = "auto", **kw: Any
    ) -> AsyncIterator[tuple[str, str]]:
        """Match the real LLM's structured stream: only 'content' events."""
        self.seen_messages.append([dict(m) for m in messages])
        text = self._pop()
        for idx in range(0, len(text), 12):
            yield "content", text[idx : idx + 12]

    def _pop(self) -> str:
        if not self._responses:
            raise RuntimeError("FakeLLM ran out of scripted responses")
        self._calls += 1
        return self._responses.pop(0)


class FakeRoutingConfig:
    def __init__(self) -> None:
        self.default_brain = "e4b"
        self.rules: list[Any] = []


class FakeBrainConfig:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = f"fake-{name}"
        self.endpoint = "http://127.0.0.1:5006/v1"
        self.api_key = "sk-test"
        self.thinking = False
        self.reasoning_budget = None


class FakeConfig:
    def __init__(self) -> None:
        self.routing = FakeRoutingConfig()
        self.brains = {"e4b": FakeBrainConfig("e4b"), "a4b": FakeBrainConfig("a4b")}

    def get_brain(self, name: str) -> FakeBrainConfig:
        return self.brains.get(name, self.brains["e4b"])


@dataclass
class FakeApp:
    config: FakeConfig = field(default_factory=FakeConfig)
    signals: LexySignals = field(default_factory=LexySignals)
    event_bus: EventBus = field(default_factory=EventBus)
    hooks: HookManager = field(default_factory=HookManager)
    session_store: SessionStore = field(default_factory=SessionStore)
    llm: FakeLLM | None = None
    memory: MemoryManager | None = None
    tool_registry: ToolRegistry | None = None
    tool_caller: ToolCaller | None = None


def _build_agent(responses: list[str], *, with_weather: bool = True) -> LexyAgent:
    app = FakeApp()
    app.llm = FakeLLM(responses)
    if with_weather:
        registry = ToolRegistry()
        registry.register(
            name="get_weather",
            handler=lambda location, units="metric": {
                "location": location,
                "temperature": 9.4,
                "conditions": "clear",
            },
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
        app.tool_registry = registry
        app.tool_caller = ToolCaller(registry)
    return LexyAgent(app)  # type: ignore[arg-type]


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plain_answer_streams_live() -> None:
    """A plain answer flows through as chunks + done."""
    agent = _build_agent(
        ["Hallo Mike! Ich bin Lexy, dein lokaler KI-Assistent und helfe gerne."]
    )
    events: list[dict[str, Any]] = []
    async for event in agent.process_stream("hi", session_id="s1"):
        events.append(event)

    chunk_events = [e for e in events if e["type"] == "chunk"]
    assert chunk_events, "expected at least one chunk"
    full = "".join(e["text"] for e in chunk_events)
    assert "Hallo Mike" in full
    assert events[-1] == {"type": "done", "tools_used": []}
    # Plain answers should never emit tool_call / tool_result
    assert not any(e["type"] in ("tool_call", "tool_result") for e in events)


@pytest.mark.asyncio
async def test_tool_call_then_final_answer() -> None:
    """
    Tool-using turn: the first LLM response is a Gemma-style tool call, the
    second (after the tool result) is a plain answer. The GUI must never see
    the raw tool-call block.
    """
    tool_call_response = (
        '<|tool_call>call\n'
        '{"name": "get_weather", "arguments": {"location": "Hechthausen"}}'
    )
    final_response = (
        "Das Wetter in Hechthausen ist momentan klar bei 9,4°C. "
        "Alles Gute dabei!"
    )
    agent = _build_agent([tool_call_response, final_response])

    events: list[dict[str, Any]] = []
    async for event in agent.process_stream(
        "Wie ist das Wetter in Hechthausen?",
        session_id="s2",
    ):
        events.append(event)

    types = [e["type"] for e in events]
    # Expected ordering: tool_call → tool_result → chunks → done
    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-1] == "done"

    # No raw tool-call markers should reach the client
    chunk_text = "".join(e["text"] for e in events if e["type"] == "chunk")
    assert "<|tool_call" not in chunk_text
    assert "<tool_call>" not in chunk_text
    assert "9,4" in chunk_text  # the final answer contains the weather

    # tools_used is populated on the done event
    done_event = events[-1]
    assert done_event["tools_used"] == ["get_weather"]


@pytest.mark.asyncio
async def test_history_persists_across_turns() -> None:
    """
    Multi-turn dialog: turn 1 mentions Hechthausen, turn 2 asks a follow-up
    without the location. The injected history must contain both the user's
    first message and the assistant's first reply.
    """
    agent = _build_agent(
        [
            "Das Wetter in Hechthausen ist aktuell 9°C und klar.",
            "Heute Nacht wird es in Hechthausen auf etwa 5°C abkühlen.",
        ],
        with_weather=False,  # focus on history injection, not the tool
    )

    # Turn 1
    events1: list[dict[str, Any]] = []
    async for event in agent.process_stream(
        "Wie ist das Wetter in Hechthausen?",
        session_id="dialog",
    ):
        events1.append(event)

    assert agent._app.session_store.length("dialog") == 2  # type: ignore[attr-defined]

    # Turn 2
    events2: list[dict[str, Any]] = []
    async for event in agent.process_stream(
        "Und wie wird es heute Nacht?",
        session_id="dialog",
    ):
        events2.append(event)

    # The second LLM call must have seen the first turn in its messages list
    # (system, user1, assistant1, user2).
    second_call_messages = agent._app.llm.seen_messages[-1]  # type: ignore[union-attr]
    roles = [m["role"] for m in second_call_messages]
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles
    assistant_contents = [
        m["content"] for m in second_call_messages if m["role"] == "assistant"
    ]
    assert any("Hechthausen" in c for c in assistant_contents)
    user_contents = [m["content"] for m in second_call_messages if m["role"] == "user"]
    assert any("Wie ist das Wetter" in c for c in user_contents)
    assert any("heute Nacht" in c for c in user_contents)

    # Store now holds 4 messages
    assert agent._app.session_store.length("dialog") == 4  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_history_drops_tool_call_markers() -> None:
    """
    If an assistant message ever contains raw tool-call markers (shouldn't
    happen normally but let's be defensive), _plan must scrub them before
    replaying the history to the LLM.
    """
    agent = _build_agent(
        ["Alles klar, wird gemacht!"],
    )
    # Simulate a dirty prior turn
    agent._app.session_store.append_pair(  # type: ignore[attr-defined]
        "dirty",
        "vorige frage",
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>'
        " Die Temperatur ist 10°C.",
    )

    events: list[dict[str, Any]] = []
    async for event in agent.process_stream("danke", session_id="dirty"):
        events.append(event)

    second_messages = agent._app.llm.seen_messages[-1]  # type: ignore[union-attr]
    assistants = [m["content"] for m in second_messages if m["role"] == "assistant"]
    assert assistants, "history must include the (cleaned) assistant reply"
    assert all("<tool_call>" not in a for a in assistants)
    # The human-readable part survives
    assert any("Temperatur" in a for a in assistants)


@pytest.mark.asyncio
async def test_unknown_tool_is_filtered() -> None:
    """
    If the model emits a tool-call shape for a tool that doesn't exist,
    the look-ahead buffers silently, detect_all returns an empty list,
    and strip_tool_call scrubs the markers before flushing.
    """
    agent = _build_agent(
        [
            '<tool_call>{"name": "does_not_exist", "arguments": {}}</tool_call>'
            " hey that tool is fake"
        ]
    )
    events: list[dict[str, Any]] = []
    async for event in agent.process_stream("hi", session_id="s3"):
        events.append(event)

    chunk_text = "".join(e["text"] for e in events if e["type"] == "chunk")
    assert "<tool_call>" not in chunk_text
    assert "does_not_exist" not in chunk_text or "fake" in chunk_text
    assert events[-1]["type"] == "done"
    assert events[-1]["tools_used"] == []
