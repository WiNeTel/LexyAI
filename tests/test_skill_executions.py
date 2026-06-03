"""
Tests for Phase-P5 skill self-refinement.

* ``should_refine`` — the pure failure-threshold gate.
* ``SkillExecutionLog`` — record + recent-failures round-trip (real aiosqlite).
* ``_maybe_refine_skill`` — only fires for managed auto-skills over threshold,
  respects the cooldown, and never touches the user's own (manual) skills.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from plugins.skill_writer.skill_executions import SkillExecutionLog, should_refine
from plugins.skill_writer.skill_writer_plugin import (
    _REFINE_COOLDOWN_SECONDS,
    SkillWriterPlugin,
)


# ─── should_refine ──────────────────────────────────────────────────────────


def test_should_refine_threshold() -> None:
    assert should_refine(3, threshold=3) is True
    assert should_refine(4, threshold=3) is True
    assert should_refine(2, threshold=3) is False


def test_should_refine_disabled_when_threshold_zero() -> None:
    assert should_refine(99, threshold=0) is False


# ─── Execution log round-trip ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execution_log_records_and_reads_failures() -> None:
    db = await aiosqlite.connect(":memory:")
    log = SkillExecutionLog(db)
    await log.init_tables()

    await log.record("demo", ok=True)
    await log.record("demo", ok=False, error="boom-1")
    await log.record("demo", ok=False, error="boom-2")
    await log.record("other", ok=False, error="unrelated")

    failures = await log.recent_failures("demo", limit=5)
    assert [f["error"] for f in failures] == ["boom-2", "boom-1"]  # newest first
    # Other skills aren't mixed in.
    assert all("unrelated" not in f["error"] for f in failures)
    await db.close()


# ─── _maybe_refine_skill gating ─────────────────────────────────────────────


class _RegStub:
    def __init__(self, entry: Any) -> None:
        self._entry = entry

    async def get(self, name: str) -> Any:
        return self._entry


def _refine_plugin(entry: Any) -> SkillWriterPlugin:
    plugin = SkillWriterPlugin.__new__(SkillWriterPlugin)
    plugin._self_refine_skills = True  # noqa: SLF001
    plugin._refine_after_failures = 3  # noqa: SLF001
    plugin._refine_cooldown = {}  # noqa: SLF001
    plugin._registry = _RegStub(entry)  # type: ignore[assignment]
    plugin._exec_log = object()  # type: ignore[assignment]
    plugin._refine_skill = AsyncMock()  # type: ignore[method-assign]
    return plugin


def _entry(
    *, name: str = "demo", source: str = "auto_pattern", failure_count: int = 3
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, source=source, failure_count=failure_count, version=1, id="id1"
    )


@pytest.mark.asyncio
async def test_refine_fires_over_threshold() -> None:
    plugin = _refine_plugin(_entry(failure_count=3))
    await plugin._maybe_refine_skill("demo")
    plugin._refine_skill.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refine_skipped_below_threshold() -> None:
    plugin = _refine_plugin(_entry(failure_count=2))
    await plugin._maybe_refine_skill("demo")
    plugin._refine_skill.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refine_never_touches_manual_skills() -> None:
    plugin = _refine_plugin(_entry(source="manual", failure_count=10))
    await plugin._maybe_refine_skill("demo")
    plugin._refine_skill.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refine_respects_cooldown() -> None:
    plugin = _refine_plugin(_entry(failure_count=5))
    # Pretend a refine just happened.
    import time

    plugin._refine_cooldown["demo"] = time.time()  # noqa: SLF001
    await plugin._maybe_refine_skill("demo")
    plugin._refine_skill.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refine_disabled_does_nothing() -> None:
    plugin = _refine_plugin(_entry(failure_count=9))
    plugin._self_refine_skills = False  # noqa: SLF001
    await plugin._maybe_refine_skill("demo")
    plugin._refine_skill.assert_not_awaited()  # type: ignore[attr-defined]


def test_refine_cooldown_constant_is_sane() -> None:
    assert _REFINE_COOLDOWN_SECONDS >= 60
