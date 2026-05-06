"""
Tests for Phase-4 project persona-override plumbing.

The agent's ``_plan`` stage builds the system prompt by concatenating
``persona.rendered_system_prompt()`` with a project-specific override
block. These tests exercise that wiring without spinning up the full
LexyApp — we fake the minimal surface ``_plan`` needs (session_store,
project_store, persona, router, tool_caller, event_bus, hooks).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from lexy_core.agent.agent import LexyAgent
from lexy_core.project import Project


# ─── Minimal stand-ins for the app dependencies ─────────────────────────────


class _FakeSessionStore:
    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        self._meta = meta or {}
        self._history: list[dict[str, str]] = []

    def get_meta(self, session_id: str) -> dict[str, Any]:
        return dict(self._meta)

    def get(self, session_id: str) -> list[dict[str, str]]:
        return list(self._history)


class _FakeProjectStore:
    def __init__(self, project: Project | None) -> None:
        self._project = project

    def get(self, project_id: str) -> Project | None:
        if self._project is None:
            return None
        if project_id == self._project.id:
            return self._project
        return None


class _FakePersona:
    thinking_enabled = False

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt

    def rendered_system_prompt(self) -> str:
        return self._prompt


class _FakeRouter:
    def route(self, text: str, brain: str, **_kwargs: Any) -> tuple[str, str]:
        # ``**_kwargs`` swallows new keyword-only options the real router
        # gains over time (e.g. has_images for vision-routing) so this
        # fake doesn't have to track every signature change.
        return ("e4b", "test")


class _FakeHooks:
    async def execute_modifying(self, name: str, ctx: dict[str, Any]) -> dict[str, Any]:
        return ctx

    async def execute_void(self, name: str, ctx: dict[str, Any]) -> None:
        return None


class _FakeEventBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))


def _build_agent(
    *,
    project: Project | None,
    meta: dict[str, Any] | None = None,
    persona_prompt: str = "Base persona prompt.",
) -> tuple[LexyAgent, SimpleNamespace]:
    app = SimpleNamespace()
    app.session_store = _FakeSessionStore(meta=meta)
    app.project_store = _FakeProjectStore(project)
    app.persona = _FakePersona(persona_prompt)
    app.hooks = _FakeHooks()
    app.event_bus = _FakeEventBus()
    app.memory = None
    app.tool_caller = None
    app.llm = None
    # BrainRouter expects a routing config but LexyAgent wraps it in
    # ``BrainRouter(app.config.routing)`` during __init__. We replace the
    # router attribute post-construction to sidestep that.
    app.config = SimpleNamespace(
        routing=SimpleNamespace(
            default_brain="e4b", rules=[], fallback=SimpleNamespace(
                cloud_enabled=False, provider="google"
            ),
        )
    )
    agent = LexyAgent(app)  # type: ignore[arg-type]
    agent._router = _FakeRouter()  # type: ignore[assignment]
    return agent, app


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── _think attaches project to ctx ─────────────────────────────────────────


def test_think_attaches_project_to_ctx() -> None:
    project = Project(id="proj-1", name="Spielefirma", persona_override="")
    agent, app = _build_agent(
        project=project, meta={"project_id": "proj-1"}
    )
    ctx: dict[str, Any] = {"text": "hi", "session_id": "s"}
    _run(agent._think(ctx))
    assert ctx["project_id"] == "proj-1"
    assert ctx["project"] is project


def test_think_defaults_when_session_has_no_project() -> None:
    """Sessions with no ``project_id`` in meta → default id on ctx."""
    agent, app = _build_agent(project=None, meta={})
    ctx: dict[str, Any] = {"text": "hi", "session_id": "s"}
    _run(agent._think(ctx))
    assert ctx["project_id"] == "default"
    assert ctx["project"] is None  # store has no default installed


# ─── _plan appends override when configured ─────────────────────────────────


def test_plan_appends_persona_override() -> None:
    project = Project(
        id="proj-1",
        name="Spielefirma",
        persona_override="Wir arbeiten an einem neuen Indie-Spiel.",
    )
    agent, app = _build_agent(
        project=project, meta={"project_id": "proj-1"}
    )
    ctx: dict[str, Any] = {
        "text": "Status?",
        "session_id": "s",
        "project": project,
        "project_id": "proj-1",
    }
    messages = _run(agent._plan(ctx))
    system = messages[0]["content"]
    assert "Base persona prompt." in system
    assert "## Projekt-Kontext: Spielefirma" in system
    assert "Wir arbeiten an einem neuen Indie-Spiel." in system


def test_plan_omits_override_when_empty() -> None:
    project = Project(
        id="proj-2", name="Leer", persona_override=""
    )
    agent, app = _build_agent(
        project=project, meta={"project_id": "proj-2"}
    )
    ctx: dict[str, Any] = {
        "text": "hi",
        "session_id": "s",
        "project": project,
        "project_id": "proj-2",
    }
    messages = _run(agent._plan(ctx))
    system = messages[0]["content"]
    assert "## Projekt-Kontext" not in system


def test_plan_omits_override_with_whitespace_only() -> None:
    project = Project(
        id="proj-3", name="Whitespace", persona_override="    \n  \t  "
    )
    agent, app = _build_agent(
        project=project, meta={"project_id": "proj-3"}
    )
    ctx: dict[str, Any] = {
        "text": "hi",
        "session_id": "s",
        "project": project,
        "project_id": "proj-3",
    }
    messages = _run(agent._plan(ctx))
    system = messages[0]["content"]
    assert "## Projekt-Kontext" not in system


def test_plan_works_when_project_missing() -> None:
    """A missing project (deleted mid-session?) must not crash _plan."""
    agent, app = _build_agent(project=None, meta={})
    ctx: dict[str, Any] = {
        "text": "hi",
        "session_id": "s",
        "project": None,
        "project_id": "default",
    }
    messages = _run(agent._plan(ctx))
    system = messages[0]["content"]
    assert "Base persona prompt." in system
    assert "## Projekt-Kontext" not in system
