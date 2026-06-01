"""Tests for per-RP-session world-state persistence (world.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from plugins.character_chat import rp_world_tools as rwt
from plugins.character_chat.rp_session_store import RPSessionContainer


class _FakeMemory:
    """Minimal MemoryBackend stub — RPSessionContainer only needs these."""

    async def ensure_collection(self, name: str) -> None:
        return None

    async def delete_collection(self, name: str) -> None:
        return None


@pytest.mark.asyncio
async def test_new_session_has_empty_world(tmp_path: Path) -> None:
    ct = await RPSessionContainer.create(tmp_path, "sid1", _FakeMemory())
    try:
        assert await ct.get_world() == {}
        assert (ct.folder / "world.json").exists()
    finally:
        await ct.close()


@pytest.mark.asyncio
async def test_set_get_world_roundtrip(tmp_path: Path) -> None:
    ct = await RPSessionContainer.create(tmp_path, "sid2", _FakeMemory())
    try:
        world = rwt.define_need(
            {},
            entity="baby",
            attribute="hunger",
            value=40.0,
            rate_per_minute=2.5,
            thresholds=[{"at": 70.0, "need": "feed_baby"}],
            minutes_per_tick=2.0,
        )
        await ct.set_world(world)
        got = await ct.get_world()
        assert rwt.snapshot(got) == {"baby": {"hunger": 40.0}}
        assert len(got["specs"]) == 1
    finally:
        await ct.close()


@pytest.mark.asyncio
async def test_world_survives_reopen(tmp_path: Path) -> None:
    ct = await RPSessionContainer.create(tmp_path, "sid3", _FakeMemory())
    world = rwt.define_need(
        {},
        entity="baby",
        attribute="hunger",
        value=55.0,
        rate_per_minute=1.0,
        thresholds=[{"at": 70.0, "need": "feed_baby"}],
        minutes_per_tick=1.0,
    )
    await ct.set_world(world)
    await ct.close()

    reopened = await RPSessionContainer.open(tmp_path, "sid3", _FakeMemory())
    try:
        got = await reopened.get_world()
        assert rwt.snapshot(got) == {"baby": {"hunger": 55.0}}
    finally:
        await reopened.close()
