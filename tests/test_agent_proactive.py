"""
Tests for :meth:`LexyAgent.process_proactive` — the entry point the
scheduler uses to make Lexy speak unprompted.

We fake the minimal LexyApp surface (llm, persona, session_store,
project_store, ws_server, event_bus, hooks) so these tests exercise
``process_proactive`` end-to-end without ChromaDB or the real gateway.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from lexy_core.agent.agent import LexyAgent
from lexy_core.project import Project


# ─── Fakes ──────────────────────────────────────────────────────────────────


class _FakeSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self.appended_assistants: list[tuple[str, str]] = []

    def get(self, session_id: str) -> list[dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def get_meta(self, session_id: str) -> dict[str, Any]:
        return dict(self._meta.get(session_id, {}))

    def set_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        self._meta[session_id] = dict(meta)

    def append_assistant(self, session_id: str, assistant_text: str) -> None:
        self.appended_assistants.append((session_id, assistant_text))
        self._sessions.setdefault(session_id, []).append(
            {"role": "assistant", "content": assistant_text}
        )


class _FakeProjectStore:
    def __init__(self, projects: dict[str, Project]) -> None:
        self._projects = projects

    def get(self, project_id: str) -> Project | None:
        return self._projects.get(project_id)


class _FakePersona:
    thinking_enabled = False

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt

    def rendered_system_prompt(self) -> str:
        return self._prompt


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self, messages: list[dict[str, str]], brain: str = "auto", **kwargs: Any
    ) -> str:
        self.calls.append({"messages": messages, "brain": brain})
        return self._response


class _FakeWSServer:
    def __init__(self) -> None:
        self.broadcasts: list[dict[str, Any]] = []

    async def broadcast(self, data: dict[str, Any]) -> None:
        self.broadcasts.append(dict(data))


class _FakeEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, dict(payload)))


class _FakeHooks:
    async def execute_modifying(self, name: str, ctx: dict[str, Any]) -> dict[str, Any]:
        return ctx

    async def execute_void(self, name: str, ctx: dict[str, Any]) -> None:
        return None


def _build_agent(
    *,
    llm_response: str = "Hey Mike, das Wasser kocht.",
    persona_prompt: str = "Du bist Lexy.",
    projects: dict[str, Project] | None = None,
    session_meta: dict[str, dict[str, Any]] | None = None,
    sessions: dict[str, list[dict[str, str]]] | None = None,
) -> tuple[LexyAgent, SimpleNamespace]:
    app = SimpleNamespace()
    app.session_store = _FakeSessionStore()
    for sid, meta in (session_meta or {}).items():
        app.session_store.set_meta(sid, meta)
    for sid, msgs in (sessions or {}).items():
        app.session_store._sessions[sid] = list(msgs)
    app.project_store = _FakeProjectStore(projects or {})
    app.persona = _FakePersona(persona_prompt)
    app.llm = _FakeLLM(llm_response)
    app.ws_server = _FakeWSServer()
    app.event_bus = _FakeEventBus()
    app.hooks = _FakeHooks()
    app.tool_caller = None
    app.memory = None
    app.signals = SimpleNamespace(update=lambda **kw: None)
    app.config = SimpleNamespace(
        routing=SimpleNamespace(
            default_brain="e4b",
            rules=[],
            fallback=SimpleNamespace(cloud_enabled=False, provider="google"),
        )
    )
    agent = LexyAgent(app)  # type: ignore[arg-type]
    return agent, app


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_process_proactive_appends_assistant_and_broadcasts() -> None:
    agent, app = _build_agent(llm_response="Ich denk grad an dich, Mike.")
    result = _run(
        agent.process_proactive(
            {
                "session_id": "s1",
                "text": "Erinnere Mike ans Training",
                "label": "Workout",
                "from": "scheduler",
            }
        )
    )
    assert result["ok"] is True
    assert result["text"] == "Ich denk grad an dich, Mike."
    assert app.session_store.appended_assistants == [
        ("s1", "Ich denk grad an dich, Mike.")
    ]
    # Frontend gets the message.
    assert app.ws_server.broadcasts
    ev = app.ws_server.broadcasts[0]
    assert ev["type"] == "proactive_message"
    assert ev["session_id"] == "s1"
    assert ev["from"] == "scheduler"
    assert ev["label"] == "Workout"

    # EventBus is notified too.
    names = [n for n, _ in app.event_bus.events]
    assert "core.proactive_message" in names


def test_process_proactive_uses_persona_and_project_override() -> None:
    project = Project(
        id="p1",
        name="Spielefirma",
        persona_override="Wir arbeiten am neuen Indie-Spiel.",
    )
    agent, app = _build_agent(
        projects={"p1": project},
        session_meta={"s1": {"project_id": "p1"}},
    )
    _run(
        agent.process_proactive(
            {"session_id": "s1", "text": "hi", "from": "scheduler"}
        )
    )
    # System prompt should include both persona + project override.
    sent = app.llm.calls[0]["messages"]
    system = sent[0]["content"]
    assert "Du bist Lexy." in system
    assert "## Projekt-Kontext: Spielefirma" in system
    assert "Wir arbeiten am neuen Indie-Spiel." in system
    # The trigger nudge now lives as the FINAL user message (llama.cpp
    # rejects "enable_thinking" + trailing assistant). It must reach the
    # prompt — just in the user role instead of the system role.
    assert sent[-1]["role"] == "user"
    assert "Interner Trigger" in sent[-1]["content"]
    assert "scheduler" in sent[-1]["content"]


def test_process_proactive_omits_project_block_when_missing() -> None:
    agent, app = _build_agent()
    _run(
        agent.process_proactive(
            {"session_id": "unknown", "text": "hi"}
        )
    )
    sent = app.llm.calls[0]["messages"]
    system = sent[0]["content"]
    assert "## Projekt-Kontext" not in system
    # Still has the trigger nudge — but at the user end, not system.
    assert sent[-1]["role"] == "user"
    assert "Interner Trigger" in sent[-1]["content"]


def test_process_proactive_uses_recent_history_slice() -> None:
    agent, app = _build_agent(
        sessions={
            "s1": [
                {"role": "user", "content": f"msg {i}"} for i in range(20)
            ]
        }
    )
    _run(
        agent.process_proactive({"session_id": "s1", "text": "trigger"})
    )
    sent = app.llm.calls[0]["messages"]
    # system + up to 8 recent turns + 1 trailing trigger-user message
    assert len(sent) <= 1 + 8 + 1
    assert sent[0]["role"] == "system"
    assert sent[-1]["role"] == "user"  # enforced by the prefill fix
    # Most recent history entry should be included (before the trigger).
    contents = [m["content"] for m in sent[1:-1]]
    assert "msg 19" in contents


def test_process_proactive_refuses_empty_response() -> None:
    agent, app = _build_agent(llm_response="   ")
    result = _run(
        agent.process_proactive({"session_id": "s1", "text": "t"})
    )
    assert result["ok"] is False
    assert result["reason"] == "empty_response"
    # Nothing persisted or broadcast.
    assert app.session_store.appended_assistants == []
    assert app.ws_server.broadcasts == []


def test_process_proactive_handles_llm_failure() -> None:
    agent, app = _build_agent()
    app.llm.chat = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]
    result = _run(
        agent.process_proactive({"session_id": "s1", "text": "t"})
    )
    assert result["ok"] is False
    assert "boom" in result["reason"]


def test_process_proactive_handles_missing_llm() -> None:
    agent, app = _build_agent()
    app.llm = None  # type: ignore[assignment]
    result = _run(
        agent.process_proactive({"session_id": "s1", "text": "t"})
    )
    assert result["ok"] is False
    assert result["reason"] == "llm_unavailable"


def test_process_proactive_trigger_text_appears_in_prompt() -> None:
    agent, app = _build_agent()
    _run(
        agent.process_proactive(
            {"session_id": "s1", "text": "Frag Mike ob das Training geklappt hat"}
        )
    )
    sent = app.llm.calls[0]["messages"]
    # Trigger text now lives in the trailing user message.
    assert sent[-1]["role"] == "user"
    assert "Frag Mike ob das Training geklappt hat" in sent[-1]["content"]


def test_process_proactive_without_trigger_text_still_works() -> None:
    agent, app = _build_agent()
    result = _run(
        agent.process_proactive(
            {"session_id": "s1", "from": "impulse"}
        )
    )
    assert result["ok"] is True
