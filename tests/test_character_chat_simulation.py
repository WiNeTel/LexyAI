"""Tests for the autonomous simulation mode in character_chat.

Covers only the new code paths introduced for the sim:

* ``_tool_start_simulation`` registers a recurring scheduler timer
* ``_tool_stop_simulation`` cancels the timer and clears state
* ``_tool_simulation_status`` reports running state
* ``_run_autonomous_tick`` respects the pulse cooldown
* ``_run_autonomous_tick`` picks Lexy vs character based on probability

The existing 732 tests cover the surrounding plumbing (pulse timers,
orchestrator, agent_proactive). We don't re-verify those here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from plugins.character_chat.character_chat_plugin import (
    CharacterChatPlugin,
)


class _FakeAPI:
    """Minimal PluginAPI stand-in for testing the sim tools."""

    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.broadcasts: list[dict[str, Any]] = []
        self.proactive_calls: list[dict[str, Any]] = []
        self.timer_id_counter = 0

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.tool_calls.append((name, dict(args)))
        if name == "set_recurring":
            self.timer_id_counter += 1
            return {
                "ok": True,
                "data": {"id": f"timer{self.timer_id_counter}"},
            }
        if name == "cancel_timer":
            return {"ok": True}
        return {"ok": True}

    async def ws_broadcast(self, payload: dict[str, Any]) -> None:
        self.broadcasts.append(dict(payload))

    async def agent_proactive(
        self, session_id: str, prompt: str, label: str = ""
    ) -> bool:
        self.proactive_calls.append(
            {"session_id": session_id, "prompt": prompt, "label": label}
        )
        return True


def _build_plugin() -> CharacterChatPlugin:
    """Create a plugin with a fake API and enough state to run the sim tools."""
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin.api = _FakeAPI()
    plugin._store = None  # tools check for None and short-circuit safely
    plugin._orchestrator = None
    plugin._pulse_cooldowns = {}
    plugin._pulse_cooldown_seconds = 600.0
    plugin._simulation_timers = {}
    plugin._simulation_default_interval = 3
    plugin._lexy_turn_probability = 0.3
    return plugin


# ─── start / stop / status ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_simulation_registers_recurring_timer() -> None:
    plugin = _build_plugin()
    result = await plugin._tool_start_simulation(
        session_id="sess-1", interval_minutes=5
    )
    assert result["ok"] is True
    assert result["timer_id"] == "timer1"
    assert plugin._simulation_timers["sess-1"] == "timer1"

    # Scheduler was called with set_recurring + autonomous_sim action type.
    api = plugin.api  # type: ignore[assignment]
    name, args = api.tool_calls[0]  # type: ignore[attr-defined]
    assert name == "set_recurring"
    assert args["action_type"] == "autonomous_sim"
    assert args["pattern"] == "every 5m"
    assert args["action_payload"]["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_start_simulation_cancels_existing_timer_first() -> None:
    """Calling start twice for the same session must not leak timers."""
    plugin = _build_plugin()
    await plugin._tool_start_simulation(session_id="sess-1", interval_minutes=3)
    await plugin._tool_start_simulation(session_id="sess-1", interval_minutes=5)

    api = plugin.api  # type: ignore[assignment]
    # First call: set_recurring. Second call: cancel_timer (for old), then set_recurring (new).
    names = [t[0] for t in api.tool_calls]  # type: ignore[attr-defined]
    assert names == ["set_recurring", "cancel_timer", "set_recurring"]
    assert plugin._simulation_timers["sess-1"] == "timer2"


@pytest.mark.asyncio
async def test_start_simulation_clamps_interval_to_range() -> None:
    plugin = _build_plugin()
    # Too small → clamp to 1
    result = await plugin._tool_start_simulation(
        session_id="sess-1", interval_minutes=0
    )
    assert result["interval_minutes"] == 1
    # Too large → clamp to 15
    await plugin._tool_start_simulation(session_id="sess-2", interval_minutes=999)
    api = plugin.api  # type: ignore[assignment]
    # Last set_recurring should be "every 15m"
    last_set = [c for c in api.tool_calls if c[0] == "set_recurring"][-1]  # type: ignore[attr-defined]
    assert last_set[1]["pattern"] == "every 15m"


@pytest.mark.asyncio
async def test_stop_simulation_cancels_and_clears() -> None:
    plugin = _build_plugin()
    await plugin._tool_start_simulation(session_id="sess-1", interval_minutes=3)
    result = await plugin._tool_stop_simulation(session_id="sess-1")
    assert result["ok"] is True
    assert result["was_running"] is True
    assert "sess-1" not in plugin._simulation_timers


@pytest.mark.asyncio
async def test_stop_simulation_when_not_running_is_noop() -> None:
    plugin = _build_plugin()
    result = await plugin._tool_stop_simulation(session_id="nothing-here")
    assert result["ok"] is True
    assert result["was_running"] is False


@pytest.mark.asyncio
async def test_simulation_status_reports_running_state() -> None:
    plugin = _build_plugin()
    # Before start — not running
    s1 = await plugin._tool_simulation_status(session_id="sess-1")
    assert s1["running"] is False

    await plugin._tool_start_simulation(session_id="sess-1", interval_minutes=3)
    s2 = await plugin._tool_simulation_status(session_id="sess-1")
    assert s2["running"] is True
    assert s2["timer_id"] == "timer1"


# ─── _run_autonomous_tick ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_autonomous_tick_respects_pulse_cooldown() -> None:
    """If a pulse round just fired, the sim tick should skip."""
    plugin = _build_plugin()
    # Pretend a pulse round fired 10 seconds ago
    import time as _time
    plugin._pulse_cooldowns["sess-1"] = _time.time() - 10
    # Cooldown is 600s → 10s ago is well within the window
    await plugin._run_autonomous_tick("sess-1")

    api = plugin.api  # type: ignore[assignment]
    # No agent_proactive, no tool calls — it was debounced.
    assert api.proactive_calls == []  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_autonomous_tick_lexy_path_calls_agent_proactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With lexy_turn_probability=1.0 and no characters, Lexy always speaks."""
    plugin = _build_plugin()
    plugin._lexy_turn_probability = 1.0  # Lexy always picked

    # No _store.list_in_session → empty character list → Lexy forced anyway
    # Need to mock _get_session_state since _store is None.
    async def fake_get_state(sid: str) -> dict[str, Any]:
        return {"character_mode": 2, "scene": "Wohnzimmer", "updated_at": 0}

    monkeypatch.setattr(plugin, "_get_session_state", fake_get_state)

    # _store.list_in_session would fail if called — mock it too
    class _FakeStore:
        async def list_in_session(self, sid: str) -> list:
            return []

    plugin._store = _FakeStore()  # type: ignore[assignment]

    await plugin._run_autonomous_tick("sess-1")

    api = plugin.api  # type: ignore[assignment]
    assert len(api.proactive_calls) == 1  # type: ignore[attr-defined]
    call = api.proactive_calls[0]  # type: ignore[attr-defined]
    assert call["session_id"] == "sess-1"
    assert "Wohnzimmer" in call["prompt"]
    assert call["label"] == "autonomous_sim:lexy"


@pytest.mark.asyncio
async def test_autonomous_tick_does_not_self_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two back-to-back sim ticks must both reach the LLM.

    Regression: the sim used to write its own timestamp into the shared
    ``_pulse_cooldowns`` dict on every successful tick. With the default
    cooldown (600s) and a user-configured 2-minute interval, this blocked
    every follow-up tick because ``now - last_round < 600`` was always
    true. Only 1 in 5 ticks made it through to the LLM.

    The sim's own ``interval_minutes`` is the rate limit — we only READ
    the cooldown (to respect a recent pulse round), we don't write it.
    """
    plugin = _build_plugin()
    plugin._lexy_turn_probability = 1.0  # Lexy always, keeps test simple
    plugin._pulse_cooldown_seconds = 600.0  # default 10 min

    async def fake_get_state(sid: str) -> dict[str, Any]:
        return {"character_mode": 2, "scene": "Küche", "updated_at": 0}
    monkeypatch.setattr(plugin, "_get_session_state", fake_get_state)

    class _FakeStore:
        async def list_in_session(self, sid: str) -> list:
            return []
    plugin._store = _FakeStore()  # type: ignore[assignment]

    # Two ticks in immediate succession (simulating 2 ticks of a 2-min
    # timer with a 10-min pulse cooldown).
    await plugin._run_autonomous_tick("sess-1")
    await plugin._run_autonomous_tick("sess-1")

    api = plugin.api  # type: ignore[assignment]
    # BOTH ticks must have reached agent_proactive.
    assert len(api.proactive_calls) == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_autonomous_tick_still_respects_recent_pulse_round() -> None:
    """A real pulse round 10s ago must still debounce the next sim tick.

    Pairs with ``test_autonomous_tick_does_not_self_debounce`` — together
    they pin the new semantics: READ the cooldown (pulse protection),
    don't WRITE it (no self-block).
    """
    plugin = _build_plugin()
    plugin._lexy_turn_probability = 1.0
    import time as _time
    plugin._pulse_cooldowns["sess-1"] = _time.time() - 10  # 10s ago
    # pulse_cooldown_seconds defaults to 600 in _build_plugin → 10s < 600s

    await plugin._run_autonomous_tick("sess-1")

    api = plugin.api  # type: ignore[assignment]
    assert api.proactive_calls == []  # type: ignore[attr-defined]
