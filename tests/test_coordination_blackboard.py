"""Tests for the coordination Blackboard (shared board: posts + facts)."""

from __future__ import annotations

import aiosqlite
import pytest

from lexy_core.coordination import Blackboard


async def _board() -> tuple[Blackboard, aiosqlite.Connection]:
    db = await aiosqlite.connect(":memory:")
    bb = Blackboard(db)
    await bb.init_tables()
    return bb, db


@pytest.mark.asyncio
async def test_post_and_read_roundtrip() -> None:
    bb, db = await _board()
    try:
        pid = await bb.post("scene1", "mira", "finding", "Strand ist leer", {"x": 1})
        assert isinstance(pid, str) and len(pid) == 12

        posts = await bb.read("scene1")
        assert len(posts) == 1
        assert posts[0]["author"] == "mira"
        assert posts[0]["kind"] == "finding"
        assert posts[0]["body"] == "Strand ist leer"
        assert posts[0]["meta"] == {"x": 1}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_scope_isolation() -> None:
    bb, db = await _board()
    try:
        await bb.post("scene_a", "a", "comment", "hallo")
        await bb.post("scene_b", "b", "comment", "welt")

        a_posts = await bb.read("scene_a")
        b_posts = await bb.read("scene_b")
        assert len(a_posts) == 1 and a_posts[0]["body"] == "hallo"
        assert len(b_posts) == 1 and b_posts[0]["body"] == "welt"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_read_filters_by_kind_dead_end_register() -> None:
    bb, db = await _board()
    try:
        await bb.post("s", "team1", "finding", "Idee A")
        await bb.post("s", "team1", "dead_end", "Ansatz X führt ins Leere")
        await bb.post("s", "team2", "dead_end", "Ansatz Y auch")

        dead_ends = await bb.read("s", kinds=["dead_end"])
        assert len(dead_ends) == 2
        assert all(p["kind"] == "dead_end" for p in dead_ends)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_read_since_filter() -> None:
    bb, db = await _board()
    try:
        await bb.post("s", "a", "comment", "first")
        all_posts = await bb.read("s")
        cutoff = all_posts[0]["created_at"]
        await bb.post("s", "a", "comment", "second")

        newer = await bb.read("s", since=cutoff)
        assert len(newer) == 1
        assert newer[0]["body"] == "second"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_facts_set_get_and_upsert() -> None:
    bb, db = await _board()
    try:
        assert await bb.get_fact("s", "missing", default="x") == "x"

        await bb.set_fact("s", "decision", {"plan": "feed_baby", "urgency": 2})
        got = await bb.get_fact("s", "decision")
        assert got == {"plan": "feed_baby", "urgency": 2}

        # upsert overwrites
        await bb.set_fact("s", "decision", "done")
        assert await bb.get_fact("s", "decision") == "done"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_snapshot_returns_all_facts_for_scope() -> None:
    bb, db = await _board()
    try:
        await bb.set_fact("s", "hunger", 72)
        await bb.set_fact("s", "mood", "anxious")
        await bb.set_fact("other", "hunger", 0)

        snap = await bb.snapshot("s")
        assert snap == {"hunger": 72, "mood": "anxious"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unknown_kind_is_stored_but_allowed() -> None:
    bb, db = await _board()
    try:
        # Forgiving: a non-canonical kind is still persisted.
        await bb.post("s", "a", "weird_kind", "body")
        posts = await bb.read("s")
        assert len(posts) == 1 and posts[0]["kind"] == "weird_kind"
    finally:
        await db.close()
