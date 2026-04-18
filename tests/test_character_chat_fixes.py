"""Tests for the post-Phase-3 fixes in character_chat.

Covers only the NEW code added in this pass:

* ``CharacterCard.voice`` field round-trips through SQL row + to/from
* ``CharacterStore.init_schema`` applies the ``voice`` migration on an
  existing DB that was created without it
* Voice propagates through spawn/update updates

The orchestrator behaviour (sequential prompting, [PASS], @mention) is
already covered by ``tests/test_group_turn_sequential.py`` and didn't
change, so we don't repeat those scenarios here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.character_store import CharacterStore


# ─── CharacterCard.voice ─────────────────────────────────────────────────────


def test_voice_defaults_to_empty_string() -> None:
    card = CharacterCard(name="Lexy")
    assert card.voice == ""


def test_voice_roundtrips_through_to_row_from_row() -> None:
    card = CharacterCard(name="Luna", voice="luna_cosy")
    row = card.to_row()
    assert row["voice"] == "luna_cosy"
    restored = CharacterCard.from_row(row)
    assert restored.voice == "luna_cosy"


def test_from_row_tolerates_missing_voice_column() -> None:
    """Rows from old databases won't have the ``voice`` key."""
    row = {
        "id": "x",
        "name": "Old",
        "persona": "",
        "greeting": "",
        "scenario": "",
        "example_dialog": "",
        "avatar": "",
        "color": "#7aa2f7",
        "age_stage": "adult",
        "relationships": "{}",
        "tags": "[]",
        "active_sessions": "[]",
        "proactive_pulse_pattern": "",
        "proactive_pulse_prompt": "",
        "archived": 0,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    card = CharacterCard.from_row(row)
    assert card.voice == ""


# ─── Store migration ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_init_schema_adds_voice_column_to_legacy_db(
    tmp_path: Path,
) -> None:
    """Legacy DB built before the voice column exists gets migrated in place."""
    db_path = tmp_path / "legacy.db"
    db = await aiosqlite.connect(str(db_path))
    try:
        # Build the pre-voice schema by hand.
        await db.execute(
            """
            CREATE TABLE characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                persona TEXT DEFAULT '',
                greeting TEXT DEFAULT '',
                scenario TEXT DEFAULT '',
                example_dialog TEXT DEFAULT '',
                avatar TEXT DEFAULT '',
                color TEXT DEFAULT '#7aa2f7',
                age_stage TEXT DEFAULT 'adult',
                relationships TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                active_sessions TEXT DEFAULT '[]',
                proactive_pulse_pattern TEXT DEFAULT '',
                proactive_pulse_prompt TEXT DEFAULT '',
                archived INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        await db.execute(
            "INSERT INTO characters (id, name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("legacy1", "OldCharacter", 1.0, 1.0),
        )
        await db.commit()

        # Now apply the store's init_schema — must add `voice` without
        # dropping the existing row.
        store = CharacterStore(db)
        await store.init_schema()

        cursor = await db.execute("PRAGMA table_info(characters)")
        cols = {row[1] for row in await cursor.fetchall()}
        await cursor.close()
        assert "voice" in cols

        # Legacy row still there, with voice defaulted to ""
        card = await store.get("legacy1")
        assert card is not None
        assert card.name == "OldCharacter"
        assert card.voice == ""
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_init_schema_twice_is_noop(tmp_path: Path) -> None:
    """Applying migrations twice must not re-add columns or error."""
    db_path = tmp_path / "fresh.db"
    db = await aiosqlite.connect(str(db_path))
    try:
        store = CharacterStore(db)
        await store.init_schema()
        # Second call should be a no-op (idempotent).
        await store.init_schema()
        cursor = await db.execute("PRAGMA table_info(characters)")
        cols = [row[1] for row in await cursor.fetchall()]
        await cursor.close()
        # voice should appear exactly once.
        assert cols.count("voice") == 1
    finally:
        await db.close()


# ─── Voice writing via update() ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_can_change_voice(tmp_path: Path) -> None:
    db_path = tmp_path / "voice.db"
    db = await aiosqlite.connect(str(db_path))
    try:
        store = CharacterStore(db)
        await store.init_schema()
        created = await store.create(
            CharacterCard(name="Luna", voice="first_voice")
        )
        assert created.voice == "first_voice"

        updated = await store.update(created.id, voice="second_voice")
        assert updated is not None
        assert updated.voice == "second_voice"

        # Persistence check
        re_read = await store.get(created.id)
        assert re_read is not None
        assert re_read.voice == "second_voice"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_update_voice_to_empty_string_clears_it(tmp_path: Path) -> None:
    db_path = tmp_path / "clear_voice.db"
    db = await aiosqlite.connect(str(db_path))
    try:
        store = CharacterStore(db)
        await store.init_schema()
        card = await store.create(
            CharacterCard(name="Bob", voice="bob_voice")
        )
        updated = await store.update(card.id, voice="")
        assert updated is not None
        assert updated.voice == ""
    finally:
        await db.close()
