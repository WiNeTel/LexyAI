"""Smoke tests for EventBus, HookManager, LexySignals."""

from __future__ import annotations

import asyncio

import pytest

from lexy_core.events import EventBus, HookManager, LexySignals, SystemState


@pytest.mark.asyncio
async def test_event_bus_exact_match() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event):
        seen.append(event.name)

    bus.on("core.user_message", handler, source="test")
    count = await bus.emit("core.user_message", {"text": "hi"})
    assert count == 1
    assert seen == ["core.user_message"]


@pytest.mark.asyncio
async def test_event_bus_wildcard_and_source_cleanup() -> None:
    bus = EventBus()
    seen: list[str] = []

    def sync_handler(event):
        seen.append(event.name)

    bus.on("core.*", sync_handler, source="plugin_a")
    await bus.emit("core.ready")
    await bus.emit("core.shutdown")
    assert seen == ["core.ready", "core.shutdown"]

    bus.off_all("plugin_a")
    await bus.emit("core.ready")
    assert seen == ["core.ready", "core.shutdown"]


@pytest.mark.asyncio
async def test_hook_manager_modifying_pipeline() -> None:
    hooks = HookManager()

    def first(ctx):
        ctx["n"] = ctx.get("n", 0) + 1
        return ctx

    async def second(ctx):
        ctx["n"] += 10
        return ctx

    hooks.register("before_prompt_build", first, priority=10, source="a")
    hooks.register("before_prompt_build", second, priority=20, source="a")
    result = await hooks.execute_modifying("before_prompt_build", {"n": 0})
    assert result["n"] == 11


def test_signals_snapshot_and_update() -> None:
    sig = LexySignals()
    assert sig.system_state == SystemState.STARTING
    assert sig.is_ready() is False  # STARTING → not ready

    sig.update(system_state=SystemState.READY, ai_thinking=True)
    assert sig.is_ready() is True  # system_state=READY, not terminated
    assert sig.is_busy() is True  # ai_thinking flips busy

    snap = sig.get_snapshot()
    assert snap["system_state"] == "ready"
    assert snap["ai_thinking"] is True

    sig.update(terminate=True)
    assert sig.is_ready() is False  # terminate flag blocks ready
