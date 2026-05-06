"""
Hotfix tests for the three RP-isolation bugs Mike reported after
Phase 11 landed:

* **Bug 1** — Chat-Tab leaks character responses.
  Mike's words: "im normalen Chat ... geantwortet hat der Charakter
  aus dem RP Chat". The hook ``_hook_before_user_input`` used to
  trigger character rounds on any session whose ``character_sessions``
  row had ``character_mode != 0`` — but with Phase 9.12's split,
  the session's ``meta.kind`` is the authoritative answer to "is
  this an RP session". Stale rows from the pre-Phase-9.12 era
  could make Lexy answer with characters in the chat tab.

* **Bug 2** — Character memory leaks across RP sessions.
  Mike's words: "Hatte einen RP Chat mit Feuer und danach einen
  neuen, aber der Charakter hatte auf ein Feuer reagiert".
  ``_character_recall`` filtered by ``character_id`` only — so
  the same character spawning in two different RPs saw memories
  from BOTH sessions. Fix: also scope by ``session_id``.

* **Bug 3** — Deleting a session leaves character data behind.
  ``DELETE /api/v1/sessions/{id}`` only cleared the session_store
  messages, leaving ``character_turns`` rows + character_sessions
  rows + ChromaDB context items tagged with this session_id. So
  even after "deleting" an RP, a fresh session with the same char
  re-surfaced the deleted RP's memories. Fix: wipe all of them.

These tests pin the new contracts. They use unit-style fakes for
Bug 1 + 2 (no full-app boot) and a real LexyApp for Bug 3 (because
the cleanup spans gateway + plugin + memory + session_store).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp
from plugins.character_chat.character_chat_plugin import CharacterChatPlugin


# ─── Bug 1 — Chat-tab character_mode isolation ───────────────────────


@pytest.mark.asyncio
async def test_hook_skips_chat_kind_session_even_with_stale_mode() -> None:
    """Phase 9.12 guard: even if ``character_sessions`` has mode=1, a
    session whose ``meta.kind`` is "chat" must not run a round.

    Repros Mike's bug: pre-Phase-9.12 the session was attached to
    characters → character_mode=1. After the kind-field upgrade
    the session is logically a chat session again, but the stale
    DB row would have triggered character responses anyway.
    """
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin._store = MagicMock()
    plugin._orchestrator = MagicMock()
    # Session-store stub returns kind="chat".
    fake_session_store = MagicMock()
    fake_session_store.get_meta = MagicMock(
        return_value={"kind": "chat", "project_id": "default"}
    )
    plugin.api = MagicMock()
    plugin.api._app = MagicMock(session_store=fake_session_store)
    # Character_mode=1 in the (legacy) char_sessions table.
    plugin._get_session_state = AsyncMock(
        return_value={"character_mode": 1, "scene": ""}
    )

    ctx = {"session_id": "stale-chat-session", "text": "hi Lexy"}
    out = await plugin._hook_before_user_input(ctx)
    # No round was fired — agent goes through as normal.
    assert "skip_agent" not in out
    assert "_character_hybrid_round" not in out


@pytest.mark.asyncio
async def test_hook_runs_round_for_rp_kind_session() -> None:
    """Sanity: kind=rp + mode=1 still triggers the round."""
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin._store = MagicMock()
    plugin._orchestrator = MagicMock()
    fake_session_store = MagicMock()
    fake_session_store.get_meta = MagicMock(
        return_value={"kind": "rp", "project_id": "default"}
    )
    plugin.api = MagicMock()
    plugin.api._app = MagicMock(session_store=fake_session_store)
    plugin._get_session_state = AsyncMock(
        return_value={"character_mode": 1, "scene": ""}
    )
    plugin._run_round_safe = AsyncMock()

    ctx = {"session_id": "rp-session", "text": "hello chars"}
    out = await plugin._hook_before_user_input(ctx)
    # mode=1 → skip_agent True (Lexy silenced, characters answer).
    assert out.get("skip_agent") is True
    assert out.get("skip_reason") == "character_mode"


@pytest.mark.asyncio
async def test_hook_falls_back_when_session_store_missing() -> None:
    """If api._app.session_store isn't reachable (test stubs), the
    hook falls back to the legacy mode check rather than crashing."""
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin._store = MagicMock()
    plugin._orchestrator = MagicMock()
    plugin.api = MagicMock()
    plugin.api._app = MagicMock(session_store=None)  # missing
    plugin._get_session_state = AsyncMock(
        return_value={"character_mode": 0, "scene": ""}
    )

    ctx = {"session_id": "test", "text": "hi"}
    out = await plugin._hook_before_user_input(ctx)
    # No character_mode → ctx returned unchanged. No crash.
    assert "skip_agent" not in out


# ─── Bug 2 — Per-session character recall ────────────────────────────


@pytest.mark.asyncio
async def test_character_recall_includes_session_id_filter() -> None:
    """``_character_recall`` must pass ``session_id`` into
    ``api.memory_recall``'s metadata_equals so memories from another
    RP don't bleed across."""
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin.api = MagicMock()
    plugin.api.memory_recall = AsyncMock(return_value=[])

    await plugin._character_recall(
        character_id="char-anna",
        query="Was ist passiert?",
        limit=3,
        session_id="rp-session-B",
    )

    plugin.api.memory_recall.assert_awaited_once()
    kwargs = plugin.api.memory_recall.await_args.kwargs
    assert kwargs["metadata_equals"] == {
        "character_id": "char-anna",
        "session_id": "rp-session-B",
    }


@pytest.mark.asyncio
async def test_character_recall_warns_without_session_id() -> None:
    """Backwards-compat: omit session_id and the filter is char-only.
    But we log a warning so the hole gets noticed in production logs."""
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin.api = MagicMock()
    plugin.api.memory_recall = AsyncMock(return_value=[])

    await plugin._character_recall(
        character_id="char-anna", query="x", limit=3,
    )

    kwargs = plugin.api.memory_recall.await_args.kwargs
    assert kwargs["metadata_equals"] == {"character_id": "char-anna"}


# ─── Bug 3 — wipe_session_data ───────────────────────────────────────


@pytest_asyncio.fixture
async def plugin_with_db(tmp_path: Path):
    """Build a real CharacterChatPlugin with an in-memory SQLite DB
    so we can populate character_turns + character_sessions and
    verify the wipe deletes them."""
    db_path = tmp_path / "char_chat.db"
    db = await aiosqlite.connect(str(db_path))

    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin.api = MagicMock()
    plugin.api.get_db = AsyncMock(return_value=db)
    plugin._store = None  # exercising the no-store branch
    plugin._cancel_pulse_timer = AsyncMock()

    # Schema (lifted straight from the plugin's on_load).
    await db.execute(
        "CREATE TABLE character_turns ("
        "id TEXT PRIMARY KEY, session_id TEXT, character_id TEXT, "
        "character_name TEXT, round_id TEXT, order_num INTEGER, "
        "content TEXT, skipped INTEGER, trigger_kind TEXT, "
        "trigger_text TEXT, created_at REAL)"
    )
    await db.execute(
        "CREATE TABLE character_sessions ("
        "session_id TEXT PRIMARY KEY, character_mode INTEGER, "
        "scene TEXT, updated_at REAL)"
    )
    await db.commit()

    yield plugin, db
    await db.close()


@pytest.mark.asyncio
async def test_wipe_session_data_drops_turns_and_session_row(
    plugin_with_db,
) -> None:
    plugin, db = plugin_with_db

    # Seed two sessions worth of data.
    for sid, name in (
        ("session-keep", "alpha"),
        ("session-wipe", "alpha"),
        ("session-wipe", "beta"),
    ):
        await db.execute(
            "INSERT INTO character_turns "
            "(id, session_id, character_id, character_name, round_id, "
            "order_num, content, skipped, trigger_kind, trigger_text, created_at) "
            "VALUES (?, ?, 'c1', ?, 'r1', 0, 'hi', 0, 'user', '?', ?)",
            (uuid.uuid4().hex[:12], sid, name, time.time()),
        )
    await db.execute(
        "INSERT INTO character_sessions VALUES (?, ?, ?, ?)",
        ("session-keep", 1, "kitchen", time.time()),
    )
    await db.execute(
        "INSERT INTO character_sessions VALUES (?, ?, ?, ?)",
        ("session-wipe", 1, "garden", time.time()),
    )
    await db.commit()

    report = await plugin.wipe_session_data("session-wipe")

    assert report["turns"] == 2
    assert report["sessions"] == 1

    # Verify "session-wipe" is gone, "session-keep" is intact.
    async with db.execute(
        "SELECT count(*) FROM character_turns WHERE session_id = ?",
        ("session-wipe",),
    ) as cur:
        assert (await cur.fetchone())[0] == 0
    async with db.execute(
        "SELECT count(*) FROM character_turns WHERE session_id = ?",
        ("session-keep",),
    ) as cur:
        assert (await cur.fetchone())[0] == 1
    async with db.execute(
        "SELECT count(*) FROM character_sessions",
    ) as cur:
        assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_wipe_empty_session_id_is_noop(plugin_with_db) -> None:
    plugin, _ = plugin_with_db
    report = await plugin.wipe_session_data("")
    assert report == {"turns": 0, "sessions": 0, "detached": 0, "timers": 0}


# ─── Bug 3 (gateway) — DELETE /api/v1/sessions/{id} integration ──────


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def test_delete_session_returns_full_report(
    lexy_client: TestClient,
) -> None:
    """The route now returns a richer report so callers can verify
    the wipe touched every layer."""
    sid = f"test-delete-{uuid.uuid4().hex[:8]}"
    # Register a session so there's something to clear.
    reg = lexy_client.post(
        "/api/v1/sessions/register",
        json={"session_id": sid, "project_id": "default"},
    )
    assert reg.status_code == 200

    resp = lexy_client.delete(f"/api/v1/sessions/{sid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cleared"
    # Phase-11-fix shape: top-level dropped + nested report.
    assert "report" in body
    # report must have at least the 'messages' key (always present).
    assert "messages" in body["report"]
    # If memory is up, the route reports memory_items.
    if "memory_items" in body["report"]:
        assert isinstance(body["report"]["memory_items"], int)
    # If character_chat is loaded, we get its sub-report.
    if "character_chat" in body["report"]:
        cc = body["report"]["character_chat"]
        for k in ("turns", "sessions", "detached"):
            assert k in cc
