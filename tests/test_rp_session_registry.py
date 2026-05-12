"""
Tests for :class:`RPSessionRegistry` (Phase 13).

The registry is the only entry point the plugin uses to reach
session containers. These tests pin lifecycle, idempotence, and
isolation between sessions.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from plugins.character_chat.rp_session_registry import RPSessionRegistry
from plugins.character_chat.rp_session_store import RPSessionContainer

from tests.test_rp_session_container import FakeMemory


@pytest.fixture
def memory() -> FakeMemory:
    return FakeMemory()


@pytest.mark.asyncio
class TestRegistry:
    async def test_is_rp_session_after_create(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            assert reg.is_rp_session("nope") is False
            await reg.get_or_create("foo", title="Foo")
            assert reg.is_rp_session("foo") is True
        finally:
            await reg.shutdown()

    async def test_get_or_create_returns_same_handle(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            a = await reg.get_or_create("x")
            b = await reg.get_or_create("x")
            assert a is b
        finally:
            await reg.shutdown()

    async def test_get_returns_none_for_unknown(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            assert await reg.get("nada") is None
        finally:
            await reg.shutdown()

    async def test_get_opens_existing(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            ct = await reg.get_or_create("y", title="Y")
            await ct.append_message({"role": "user", "content": "Hi"})
        finally:
            await reg.shutdown()

        # New registry — should re-open from disk.
        reg2 = RPSessionRegistry(tmp_path, memory)
        try:
            ct2 = await reg2.get("y")
            assert ct2 is not None
            assert (await ct2.get_meta())["title"] == "Y"
            assert await ct2.list_messages() == [
                {"role": "user", "content": "Hi"},
            ]
        finally:
            await reg2.shutdown()

    async def test_destroy_removes_session(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            await reg.get_or_create("temp")
            assert reg.is_rp_session("temp")
            ok = await reg.destroy("temp")
            assert ok is True
            assert not reg.is_rp_session("temp")
            # Second destroy is a no-op (returns False).
            assert await reg.destroy("temp") is False
        finally:
            await reg.shutdown()

    async def test_list_session_ids(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            await reg.get_or_create("alpha")
            await reg.get_or_create("beta")
            await reg.get_or_create("gamma")
            ids = await reg.list_session_ids()
            assert set(ids) == {"alpha", "beta", "gamma"}
        finally:
            await reg.shutdown()

    async def test_concurrent_get_or_create_no_double_open(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        """If two coroutines race on the same session_id, both must
        get the same instance — not two separate containers fighting
        over the same folder."""
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            results = await asyncio.gather(
                reg.get_or_create("racey"),
                reg.get_or_create("racey"),
                reg.get_or_create("racey"),
            )
            assert results[0] is results[1] is results[2]
        finally:
            await reg.shutdown()

    async def test_session_state_isolated_across_sessions(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        """Mike's Phase 13 contract: state in session A doesn't
        bleed into session B for the same character."""
        reg = RPSessionRegistry(tmp_path, memory)
        try:
            a = await reg.get_or_create(
                "isle_a", tracked_stats={"clothing": "nackt"},
            )
            b = await reg.get_or_create(
                "isle_b", tracked_stats={"clothing": "nackt"},
            )
            # Sandra in A: state changes to shirt
            await a.update_char_state("sandra", {"clothing": "shirt"})
            # Sandra in B: still default-only — nothing leaked
            assert (await b.get_char_state("sandra")) == {}
            # And A still has the shirt update.
            assert (await a.get_char_state("sandra")) == {"clothing": "shirt"}
        finally:
            await reg.shutdown()
