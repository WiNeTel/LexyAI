"""
Tests for the Phase 7 tool-capable mini-agent loop in
``AutonomousThinkingPlugin`` — covers whitelist enforcement, max-iteration
cap, action logging, and plain-text fallback.

We use real ``ToolRegistry`` + ``ToolCaller`` instances so the pattern
detection path is exercised end-to-end. The rest of the PluginAPI is faked.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from lexy_core.tools.tool_caller import ToolCaller
from lexy_core.tools.tool_registry import ToolRegistry
from plugins.autonomous_thinking.thinking_plugin import AutonomousThinkingPlugin


# ─── Fake API ──────────────────────────────────────────────────────────


class _FakeAPI:
    """Minimal PluginAPI surface the thinking plugin uses."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        llm_responses: list[str],
        registry: ToolRegistry,
        caller: ToolCaller,
        recall_items: list[dict[str, Any]] | None = None,
    ) -> None:
        self._config = config
        self._llm_responses = list(llm_responses)
        self._registry = registry
        self._caller = caller
        self._recall_items = list(recall_items or [])

        self.llm_calls: list[dict[str, Any]] = []
        self.tool_exec: list[tuple[str, dict[str, Any]]] = []
        self.memory_stored: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.broadcasts: list[dict[str, Any]] = []

    # ── Config ──
    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    # ── Events / WS (not exercised here but plugin's on_enable calls them)
    def on_event(self, name: str, cb: Any) -> None:
        return None

    def register_ws_handler(self, msg_type: str, handler: Any) -> None:
        return None

    async def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, dict(payload)))

    async def ws_broadcast(self, data: dict[str, Any]) -> None:
        self.broadcasts.append(dict(data))

    # ── LLM ──
    async def llm_chat(
        self, messages: list[dict[str, str]], brain: str = "auto", **kwargs: Any
    ) -> str:
        self.llm_calls.append({"messages": messages, "brain": brain, **kwargs})
        if not self._llm_responses:
            return ""
        return self._llm_responses.pop(0)

    # ── Memory ──
    async def memory_recall(
        self, query: str, collection: str = "context", limit: int = 5,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return list(self._recall_items)

    async def memory_store(
        self, text: str, collection: str = "context",
        metadata: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> str:
        self.memory_stored.append(
            {"text": text, "collection": collection, "metadata": dict(metadata or {})}
        )
        return "mem-id"

    # ── Tools ──
    def get_tool_caller(self) -> ToolCaller:
        return self._caller

    def get_tool_registry(self) -> ToolRegistry:
        return self._registry

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.tool_exec.append((name, dict(args)))
        result = await self._registry.execute(name, dict(args))
        return {"ok": result.success, "data": result.data, "error": result.error}

    def list_tool_names(self) -> list[str]:
        return [s["name"] for s in self._registry.get_all_schemas()]


def _build_registry(tools: dict[str, Any]) -> ToolRegistry:
    reg = ToolRegistry()
    for name, handler in tools.items():
        reg.register(
            name=name,
            handler=handler,
            description=f"Test tool {name}",
            schema={"type": "object", "properties": {}, "required": []},
            source="test",
        )
    return reg


def _build_plugin(
    *,
    llm_responses: list[str],
    whitelist: list[str],
    tools: dict[str, Any] | None = None,
    tools_enabled: bool = True,
    max_iterations: int = 3,
    recall_items: list[dict[str, Any]] | None = None,
) -> tuple[AutonomousThinkingPlugin, _FakeAPI]:
    registry = _build_registry(tools or {})
    caller = ToolCaller(registry)
    cfg = {
        "enabled": True,
        "modes": ["daydream"],
        "tools_enabled": tools_enabled,
        "tools_max_iterations": max_iterations,
        "tools_whitelist": whitelist,
    }
    api = _FakeAPI(
        config=cfg,
        llm_responses=llm_responses,
        registry=registry,
        caller=caller,
        recall_items=recall_items,
    )
    manifest = SimpleNamespace(name="autonomous_thinking", config_defaults={})
    plugin = AutonomousThinkingPlugin(api, manifest)  # type: ignore[arg-type]
    asyncio.get_event_loop().run_until_complete(plugin.on_load())
    return plugin, api


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Tests ─────────────────────────────────────────────────────────────


def test_plain_text_response_no_tools() -> None:
    plugin, api = _build_plugin(
        llm_responses=["Der Regen heute hat was Meditatives."],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: "stored"},
    )
    result = _run(plugin._run_thought("daydream", forced=True))
    assert result["text"] == "Der Regen heute hat was Meditatives."
    assert result["actions"] == []
    assert api.tool_exec == []
    # Thought is persisted.
    assert api.memory_stored[0]["text"] == "[daydream] Der Regen heute hat was Meditatives."


def test_single_tool_call_then_final_text() -> None:
    called: list[dict[str, Any]] = []

    def mem_store(**kw: Any) -> str:
        called.append(kw)
        return "saved"

    # Turn 1: model emits tool_call. Turn 2: model returns plain text.
    responses = [
        '<tool_call>\n{"name": "memory_store", "arguments": {"text": "Mike mag Regen"}}\n</tool_call>',
        "Hab gerade notiert dass Mike Regen mag.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store"],
        tools={"memory_store": mem_store},
    )
    result = _run(plugin._run_thought("reflect", forced=True))
    assert result["text"] == "Hab gerade notiert dass Mike Regen mag."
    assert len(result["actions"]) == 1
    assert result["actions"][0]["tool"] == "memory_store"
    assert result["actions"][0]["ok"] is True
    assert called == [{"text": "Mike mag Regen"}]


def test_non_whitelisted_tool_is_skipped() -> None:
    """Tools not in the whitelist must never be executed."""
    def forbidden(**kw: Any) -> str:
        raise AssertionError("Forbidden tool was executed")

    responses = [
        '<tool_call>\n{"name": "forbidden_tool", "arguments": {}}\n</tool_call>',
        "Ok, kein Tool, nur ein Gedanke.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store"],  # forbidden_tool is NOT here
        tools={
            "memory_store": lambda **kw: "",
            "forbidden_tool": forbidden,
        },
    )
    result = _run(plugin._run_thought("daydream", forced=True))
    # No tool was executed — it was skipped for being off-whitelist.
    assert api.tool_exec == []
    # First iteration records a skipped action.
    assert any(
        a.get("skipped") and a.get("reason") == "not_whitelisted"
        for a in result["actions"]
    )


def test_unknown_tool_is_dropped_by_registry() -> None:
    """Tool names the LLM hallucinates (not registered) are silently dropped."""
    responses = [
        '<tool_call>\n{"name": "does_not_exist", "arguments": {}}\n</tool_call>',
        "Neutraler Gedanke.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store", "does_not_exist"],
        tools={"memory_store": lambda **kw: ""},
    )
    result = _run(plugin._run_thought("worry", forced=True))
    # No call happened — the registry rejected it.
    assert api.tool_exec == []
    # Final text is the 2nd-iteration response, because there were no
    # executable calls on iteration 1.
    assert result["text"] == "Neutraler Gedanke."


def test_max_iterations_cap_stops_loop() -> None:
    """If the model keeps asking for tools, we cap at ``max_iterations``
    and still return a final text."""
    call_count = {"n": 0}

    def looper(**kw: Any) -> str:
        call_count["n"] += 1
        return f"ok-{call_count['n']}"

    # With cap=2 the loop executes 2 tool calls then falls through to one
    # final plain-text LLM call (3 LLM calls total).
    responses = [
        '<tool_call>\n{"name": "memory_store", "arguments": {"n": 1}}\n</tool_call>',
        '<tool_call>\n{"name": "memory_store", "arguments": {"n": 2}}\n</tool_call>',
        "Letzter Gedanke nach dem Cap.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store"],
        tools={"memory_store": looper},
        max_iterations=2,
    )
    result = _run(plugin._run_thought("learn", forced=True))
    # Two tool executions (one per iteration), then final plain text.
    assert call_count["n"] == 2
    assert len(result["actions"]) == 2
    assert result["text"] == "Letzter Gedanke nach dem Cap."


def test_tool_failure_recorded_as_action_error() -> None:
    """An exception from a tool handler must not crash the loop; it is
    logged in ``actions`` and the loop continues."""
    def boom(**kw: Any) -> str:
        raise RuntimeError("tool exploded")

    responses = [
        '<tool_call>\n{"name": "memory_store", "arguments": {}}\n</tool_call>',
        "Gedanke ohne Tool-Ergebnis.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store"],
        tools={"memory_store": boom},
    )
    result = _run(plugin._run_thought("reflect", forced=True))
    assert result["actions"][0]["ok"] is False
    assert "tool exploded" in result["actions"][0]["error"]
    assert result["text"] == "Gedanke ohne Tool-Ergebnis."


def test_actions_persisted_in_memory_metadata() -> None:
    """The memory_store metadata must include a serialised ``actions`` log."""
    responses = [
        '<tool_call>\n{"name": "memory_store", "arguments": {"text": "x"}}\n</tool_call>',
        "Endgedanke.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: "saved"},
    )
    _run(plugin._run_thought("reflect", forced=True))
    # Last memory_store call is the plugin's own persistence step.
    # (The tool-handler's memory_store, in this fake, doesn't hit memory_stored.)
    assert api.memory_stored  # plugin persisted the final thought
    meta = api.memory_stored[-1]["metadata"]
    assert meta["mode"] == "reflect"
    assert meta["source"] == "autonomous_thinking"
    actions = json.loads(meta["actions"])
    assert any(a.get("tool") == "memory_store" and a.get("ok") for a in actions)


def test_broadcast_includes_actions() -> None:
    """The WS broadcast carries the full action log for the frontend."""
    responses = [
        '<tool_call>\n{"name": "memory_store", "arguments": {}}\n</tool_call>',
        "Final.",
    ]
    plugin, api = _build_plugin(
        llm_responses=responses,
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: "saved"},
    )
    _run(plugin._run_thought("daydream", forced=True))
    assert api.broadcasts
    bc = api.broadcasts[-1]
    assert bc["type"] == "autonomous_thought"
    assert bc["mode"] == "daydream"
    assert bc["text"] == "Final."
    assert isinstance(bc["actions"], list)
    assert bc["actions"][0]["tool"] == "memory_store"


def test_tools_disabled_config_takes_plain_path() -> None:
    """Setting ``tools_enabled=False`` skips the tool loop entirely."""
    plugin, api = _build_plugin(
        llm_responses=["Plain gedanke"],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
        tools_enabled=False,
    )
    result = _run(plugin._run_thought("daydream", forced=True))
    assert result["actions"] == []
    assert result["text"] == "Plain gedanke"
    assert api.tool_exec == []
    # Only one LLM call — no tool loop.
    assert len(api.llm_calls) == 1


def test_empty_whitelist_takes_plain_path() -> None:
    """If the whitelist matches no registered tool, we fall through to plain."""
    plugin, api = _build_plugin(
        llm_responses=["Nur Text"],
        whitelist=["this_tool_does_not_exist"],
        tools={"memory_store": lambda **kw: ""},
    )
    result = _run(plugin._run_thought("reflect", forced=True))
    assert result["text"] == "Nur Text"
    assert result["actions"] == []
    assert len(api.llm_calls) == 1


def test_system_prompt_lists_whitelisted_tools() -> None:
    """The system prompt must name whitelisted tools so the model knows
    what's callable."""
    plugin, api = _build_plugin(
        llm_responses=["Gedanke ohne Tool."],
        whitelist=["memory_store", "set_reminder"],
        tools={
            "memory_store": lambda **kw: "",
            "set_reminder": lambda **kw: "",
            "spotify_play": lambda **kw: "",  # NOT whitelisted
        },
    )
    _run(plugin._run_thought("reflect", forced=True))
    system_content = api.llm_calls[0]["messages"][0]["content"]
    assert "memory_store" in system_content
    assert "set_reminder" in system_content
    # Off-whitelist tools must not leak into the prompt.
    assert "spotify_play" not in system_content


def test_on_config_changed_applies_live() -> None:
    """``on_config_changed`` must re-read the config without a restart."""
    plugin, api = _build_plugin(
        llm_responses=[],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
    )
    assert plugin._tools_enabled is True
    _run(plugin.on_config_changed({"enabled": True, "tools_enabled": False, "tools_whitelist": []}))
    assert plugin._tools_enabled is False
    assert plugin._tools_whitelist == []


def test_empty_response_without_actions_returns_empty() -> None:
    """An empty model response and no actions means we do nothing."""
    plugin, api = _build_plugin(
        llm_responses=["   "],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
    )
    result = _run(plugin._run_thought("daydream", forced=True))
    assert result["text"] == ""
    assert result["actions"] == []
    # Nothing persisted or broadcast.
    assert api.memory_stored == []
    assert api.broadcasts == []


# ─── Lifecycle regression: the _enabled collision bug ──────────────────
#
# Historically, the plugin stored its user-facing "loop is active" flag in
# ``self._enabled`` — the SAME attribute ``BasePlugin`` uses to mean "the
# lifecycle has completed its on_enable step". When the config set
# ``enabled: true`` in ``on_load``, the plugin's own flag turned True,
# which made ``plugin.enabled`` True (via the base-class property), which
# caused ``PluginLoader._enable_plugin`` to short-circuit:
#
#     if plugin is None or plugin.enabled: return
#
# So ``on_enable`` was never called, WS handlers never registered, and the
# background loop never started — even though ``/api/v1/plugins`` happily
# reported the plugin as ``enabled: true``. These tests lock in the fix.


def test_apply_config_does_not_flip_base_enabled() -> None:
    """After ``on_load``, ``plugin.enabled`` must STILL be False.

    ``BasePlugin.enabled`` reads ``_enabled`` (the framework lifecycle
    flag). Until ``PluginLoader._enable_plugin`` sets it, the plugin is
    not 'enabled' from the loader's perspective — no matter what the
    user-facing config says.
    """
    plugin, _api = _build_plugin(
        llm_responses=[],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
    )
    # Config said enabled=True → internal loop-flag is True
    assert plugin._thinking_active is True
    # …but the framework-level BasePlugin flag is NOT True yet; only the
    # PluginLoader is allowed to set this.
    assert plugin.enabled is False, (
        "BasePlugin.enabled must remain False after on_load so "
        "PluginLoader._enable_plugin does NOT short-circuit before "
        "calling on_enable()."
    )


def test_on_enable_starts_loop_when_config_says_enabled() -> None:
    """If config.enabled=True, ``on_enable`` must actually start the loop."""
    plugin, _api = _build_plugin(
        llm_responses=[],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
    )
    assert plugin._loop_task is None  # not started yet
    _run(plugin.on_enable())
    try:
        assert plugin._loop_task is not None, (
            "on_enable must create the thinking loop task when enabled"
        )
        assert not plugin._loop_task.done()
    finally:
        _run(plugin.on_disable())


def test_on_config_changed_starts_loop_when_flipped_on() -> None:
    """Flipping ``enabled`` from False → True via the settings UI must
    start the loop WITHOUT requiring a server restart.

    Regression test: previously ``on_config_changed`` only updated the
    flag but never created the task, so toggling in the UI only "worked"
    if you also restarted."""
    plugin, _api = _build_plugin(
        llm_responses=[],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
    )
    # Start from disabled state after on_enable()
    _run(plugin.on_config_changed({"enabled": False}))
    _run(plugin.on_enable())
    try:
        assert plugin._loop_task is None  # disabled → no loop

        # Flip the setting via the live-patch hook
        _run(plugin.on_config_changed({"enabled": True}))
        assert plugin._thinking_active is True
        assert plugin._loop_task is not None
        assert not plugin._loop_task.done()

        # And flip it back off — the loop must stop
        _run(plugin.on_config_changed({"enabled": False}))
        assert plugin._thinking_active is False
        assert plugin._loop_task is None or plugin._loop_task.done()
    finally:
        _run(plugin.on_disable())


# ─── Chat-aware context builder ────────────────────────────────────────
#
# Before these tests, ``_build_messages`` used ``memory_recall`` with a
# generic "recent conversation" query — a semantic match that almost
# never surfaced the actual recent chat, so thoughts drifted into meta-
# commentary on the system prompt. The new implementation reads
# ``session_store`` chronologically with ``signals`` as the tie-breaker.


class _FakeSessionStore:
    def __init__(self, sessions: dict[str, list[dict[str, str]]],
                 meta: dict[str, dict[str, Any]] | None = None) -> None:
        self._sessions = sessions
        self._meta = meta or {}

    def get(self, session_id: str) -> list[dict[str, str]]:
        return list(self._sessions.get(session_id, []))

    def sessions_with_meta(self) -> list[tuple[str, dict[str, Any], int]]:
        return [
            (sid, dict(self._meta.get(sid, {})), len(msgs))
            for sid, msgs in self._sessions.items()
        ]


def _with_app(api: _FakeAPI, *, sessions: dict[str, list[dict[str, str]]],
              active: str = "", meta: dict[str, dict[str, Any]] | None = None) -> None:
    """Bolt a fake LexyApp onto the fake API so the plugin's chat-tail
    helper (which reads ``api._app.session_store`` + ``signals``) works."""
    api._app = SimpleNamespace(  # type: ignore[attr-defined]
        session_store=_FakeSessionStore(sessions, meta=meta),
        signals=SimpleNamespace(active_session_id=active),
    )


def test_recent_chat_tail_uses_active_session_from_signals() -> None:
    plugin, api = _build_plugin(
        llm_responses=["quiet thought"],
        whitelist=[],
        tools_enabled=False,
    )
    _with_app(
        api,
        sessions={
            "sess-a": [
                {"role": "user", "content": "Mike: Hi"},
                {"role": "assistant", "content": "Lexy: Hey"},
                {"role": "user", "content": "Mike: Heute Factorio?"},
                {"role": "assistant", "content": "Lexy: Klar"},
            ],
            "sess-b": [{"role": "user", "content": "Other convo"}],
        },
        active="sess-a",
    )

    tail = plugin._recent_chat_tail(limit=3)
    assert [m["content"] for m in tail] == [
        "Lexy: Hey",
        "Mike: Heute Factorio?",
        "Lexy: Klar",
    ]


def test_recent_chat_tail_falls_back_to_newest_when_no_active() -> None:
    """No active session → pick the session with the latest updated_at."""
    plugin, api = _build_plugin(
        llm_responses=["quiet thought"],
        whitelist=[],
        tools_enabled=False,
    )
    _with_app(
        api,
        sessions={
            "sess-old": [{"role": "user", "content": "old"}],
            "sess-new": [{"role": "user", "content": "new"}],
            "sess-empty": [],  # zero-message sessions are skipped
        },
        active="",
        meta={
            "sess-old": {"updated_at": 100.0},
            "sess-new": {"updated_at": 500.0},
        },
    )

    tail = plugin._recent_chat_tail(limit=5)
    assert [m["content"] for m in tail] == ["new"]


def test_recent_chat_tail_returns_empty_without_app() -> None:
    plugin, _api = _build_plugin(
        llm_responses=["quiet thought"],
        whitelist=[],
        tools_enabled=False,
    )
    # No _with_app() → _app attr missing
    assert plugin._recent_chat_tail(limit=5) == []


def test_build_messages_surfaces_actual_chat_not_semantic_recall() -> None:
    """The built user-message must contain the chronological chat, not
    a '- recent conversation' vector-recall dump."""
    plugin, api = _build_plugin(
        llm_responses=["quiet thought"],
        whitelist=[],
        tools_enabled=False,
    )
    _with_app(
        api,
        sessions={
            "sess": [
                {"role": "user", "content": "Mike: mir ist langweilig"},
                {"role": "assistant", "content": "Lexy: komm Minecraft"},
            ],
        },
        active="sess",
    )

    messages = _run(plugin._build_messages("SYSTEM PROMPT"))
    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    assert "Letzter Chat-Verlauf" in user_content
    assert "Mike: mir ist langweilig" in user_content
    assert "Lexy: komm Minecraft" in user_content
    # No old placeholder language from the broken semantic recall path.
    assert "Recent context" not in user_content


def test_build_messages_appends_facts_blurb_when_available() -> None:
    """The facts blurb must be labelled separately so the model
    doesn't confuse it with the current chat."""
    plugin, api = _build_plugin(
        llm_responses=["quiet thought"],
        whitelist=[],
        tools_enabled=False,
        recall_items=[
            {"content": "Mike lebt in Hechthausen"},
            {"content": "Mike arbeitet an einer Spielefirma"},
        ],
    )
    _with_app(
        api,
        sessions={"sess": [{"role": "user", "content": "Hi"}]},
        active="sess",
    )

    messages = _run(plugin._build_messages("SYS"))
    user_content = messages[1]["content"]
    assert "Was du über Mike weißt" in user_content
    assert "Hechthausen" in user_content
    # Ordered: chat tail first, facts second
    assert user_content.index("Letzter Chat-Verlauf") < user_content.index(
        "Was du über Mike weißt"
    )


# ─── Observability: get_status() ───────────────────────────────────────


def test_get_status_reports_fields_the_ui_needs() -> None:
    plugin, _api = _build_plugin(
        llm_responses=[],
        whitelist=["memory_store"],
        tools={"memory_store": lambda **kw: ""},
    )
    # Force a pretend-firing state so the counters have something non-zero.
    plugin._last_tick_at = 1_700_000_000.0
    plugin._last_thought_at = 1_700_000_000.0
    plugin._last_thought_mode = "reflect"
    plugin._last_skip_reason = "idle_too_short"
    plugin._total_thoughts = 3
    plugin._total_ticks = 7
    plugin._recent_thoughts = [1_700_000_000.0 - 60.0] * 3

    status = plugin.get_status()
    # All fields the UI reads must be present with the right types.
    expected_keys = {
        "active",
        "loop_alive",
        "in_quiet_hours",
        "idle_seconds",
        "min_idle_seconds",
        "mode_interval_seconds",
        "max_thoughts_per_hour",
        "modes",
        "tools_enabled",
        "tools_whitelist_effective",
        "last_tick_at",
        "last_skip_reason",
        "last_thought_at",
        "last_thought_mode",
        "thoughts_last_hour",
        "total_thoughts",
        "total_ticks",
        "next_tick_in_seconds",
    }
    assert expected_keys.issubset(status.keys())
    assert status["last_skip_reason"] == "idle_too_short"
    assert status["last_thought_mode"] == "reflect"
    assert status["total_thoughts"] == 3
    assert status["total_ticks"] == 7


def test_get_status_next_tick_zero_when_loop_dead() -> None:
    plugin, _api = _build_plugin(
        llm_responses=[],
        whitelist=[],
        tools_enabled=False,
    )
    # Loop was never started → no task, no ETA
    status = plugin.get_status()
    assert status["loop_alive"] is False
    assert status["next_tick_in_seconds"] == 0.0


def test_persist_updates_last_thought_fields() -> None:
    """Storing a thought must bump ``_last_thought_at`` + mode so the UI
    can say 'letzter Gedanke vor 2 Min'."""
    plugin, api = _build_plugin(
        llm_responses=["Mike wirkt heute entspannt."],
        whitelist=[],
        tools_enabled=False,
    )
    assert plugin._last_thought_at == 0.0
    _run(plugin._run_thought("reflect", forced=True))
    assert plugin._last_thought_at > 0
    assert plugin._last_thought_mode == "reflect"
    assert plugin._total_thoughts == 1
