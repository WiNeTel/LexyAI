"""Tests that the real ``autonomous_thinking`` + ``scheduler`` plugins
apply a runtime config patch through ``on_config_changed`` without a restart.

The generic PATCH endpoint behaviour is covered by
``test_settings_live_patch.py``. This file covers the *plugin-specific*
reactions that the settings UI now triggers:

* autonomous_thinking — the loop must react to `mode_interval_seconds`,
  `quiet_hours`, `modes`, and the tool-whitelist without a restart.
* scheduler — `check_interval`, `max_active_timers`, `enable_impulses`.
"""

from __future__ import annotations

from typing import Any

import pytest


# ─── autonomous_thinking.on_config_changed ───────────────────────────────────


@pytest.mark.asyncio
async def test_autonomous_thinking_applies_config_live() -> None:
    """Applying a patch updates the in-memory fields immediately."""
    from plugins.autonomous_thinking.thinking_plugin import (
        AutonomousThinkingPlugin,
    )

    # Instantiate without going through the full plugin lifecycle — we only
    # need the config-application logic, not the running background loop.
    plugin = AutonomousThinkingPlugin.__new__(AutonomousThinkingPlugin)
    plugin.name = "autonomous_thinking"
    plugin._running = False
    plugin._loop_task = None
    plugin._thinking_active = False
    # Initial defaults
    plugin._apply_config({
        "enabled": False,
        "mode_interval_seconds": 600,
        "modes": ["daydream", "reflect", "learn", "worry"],
        "quiet_hours": ["23:00", "07:00"],
        "min_idle_seconds": 120,
        "max_thoughts_per_hour": 4,
        "tools_enabled": True,
        "tools_max_iterations": 3,
        "tools_whitelist": ["memory_store"],
    })
    assert plugin._mode_interval == 600
    assert list(plugin._tools_whitelist) == ["memory_store"]

    # Live patch — simulates the UI PATCH request reaching on_config_changed
    await plugin.on_config_changed({
        "enabled": False,
        "mode_interval_seconds": 120,
        "modes": ["daydream", "learn"],
        "quiet_hours": ["22:00", "06:00"],
        "min_idle_seconds": 300,
        "max_thoughts_per_hour": 10,
        "tools_enabled": False,
        "tools_max_iterations": 5,
        "tools_whitelist": ["memory_store", "set_reminder"],
    })

    # Live fields updated without restart
    assert plugin._mode_interval == 120
    assert plugin._min_idle_seconds == 300
    assert plugin._max_per_hour == 10
    assert plugin._tools_enabled is False
    assert plugin._tools_max_iterations == 5
    assert sorted(list(plugin._tools_whitelist)) == ["memory_store", "set_reminder"]
    assert "daydream" in plugin._modes
    assert "worry" not in plugin._modes


# ─── scheduler.on_config_changed ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scheduler_applies_config_live() -> None:
    from plugins.scheduler.scheduler_plugin import SchedulerPlugin

    plugin = SchedulerPlugin.__new__(SchedulerPlugin)
    plugin.name = "scheduler"
    plugin._apply_config({
        "check_interval": 5.0,
        "max_active_timers": 50,
        "db_filename": "scheduler.db",
        "enable_impulses": True,
        "impulse_min_hour": 8,
        "impulse_max_hour": 22,
    })
    assert plugin._check_interval == 5.0
    assert plugin._max_active == 50
    assert plugin._enable_impulses is True

    await plugin.on_config_changed({
        "check_interval": 2.0,
        "max_active_timers": 200,
        "db_filename": "scheduler.db",
        "enable_impulses": False,
        "impulse_min_hour": 9,
        "impulse_max_hour": 21,
    })

    assert plugin._check_interval == 2.0
    assert plugin._max_active == 200
    assert plugin._enable_impulses is False
    assert plugin._impulse_min_hour == 9
    assert plugin._impulse_max_hour == 21
