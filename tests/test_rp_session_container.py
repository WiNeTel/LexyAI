"""
Tests for the per-RP-session storage container (Phase 13).

The container is the foundation of Mike's "every session is a folder"
isolation. These tests pin:

* ``create()`` always yields an empty namespace
* turn / state / message round-trips
* ``destroy()`` actually removes the folder + Chroma collection
* memory_recall is scoped to the session's collection (cross-session
  isolation by construction)
* stat parsing handles Mike's typed input
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from plugins.character_chat.rp_session_store import (
    RPSessionContainer,
    TurnRow,
    parse_stats_input,
    serialise_stats,
)


# ─── Fake memory backend ────────────────────────────────────────────


class FakeMemory:
    """In-memory replacement for MemoryManager during tests.

    Tracks ensure / delete / store / recall calls so tests can assert
    isolation without touching real ChromaDB.
    """

    def __init__(self) -> None:
        # collection_name → list[ {id, text, metadata} ]
        self.items: dict[str, list[dict[str, Any]]] = {}
        self.ensured: list[str] = []
        self.deleted: list[str] = []

    async def ensure_collection(self, name: str) -> None:
        self.ensured.append(name)
        self.items.setdefault(name, [])

    async def delete_collection(self, name: str) -> None:
        self.deleted.append(name)
        self.items.pop(name, None)

    async def store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = uuid.uuid4().hex
        self.items.setdefault(collection, []).append(
            {"id": item_id, "text": text, "metadata": dict(metadata or {})}
        )
        return item_id

    async def recall(
        self,
        query: str,
        collection: str | None = None,
        limit: int = 5,
        project_id: str | None = None,
        metadata_equals: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if collection is None:
            return []
        rows = list(self.items.get(collection, []))
        if metadata_equals:
            rows = [
                r for r in rows
                if all(r["metadata"].get(k) == v for k, v in metadata_equals.items())
            ]
        # Simple substring match — good enough for tests.
        q = query.lower().strip()
        if q:
            rows = [r for r in rows if q in r["text"].lower()]
        return rows[:limit]


@pytest.fixture
def memory() -> FakeMemory:
    return FakeMemory()


# ─── Stat-input parsing ─────────────────────────────────────────────


class TestParseStatsInput:
    def test_basic_key_value(self) -> None:
        assert parse_stats_input("clothing=nackt; mood=ruhig") == {
            "clothing": "nackt", "mood": "ruhig",
        }

    def test_key_only(self) -> None:
        assert parse_stats_input("clothing; mood; hunger") == {
            "clothing": "", "mood": "", "hunger": "",
        }

    def test_mixed(self) -> None:
        assert parse_stats_input("Clothing=nackt; Posture=stehend; Mood; Hunger") == {
            "clothing": "nackt",
            "posture": "stehend",
            "mood": "",
            "hunger": "",
        }

    def test_normalises_to_snake_case(self) -> None:
        assert parse_stats_input("Hunger Level=satt") == {"hunger_level": "satt"}

    def test_empty_input(self) -> None:
        assert parse_stats_input("") == {}
        assert parse_stats_input("   ;;; ;  ") == {}

    def test_newlines_treated_as_separators(self) -> None:
        assert parse_stats_input("clothing=nackt\nmood=ruhig") == {
            "clothing": "nackt", "mood": "ruhig",
        }

    def test_serialise_roundtrip(self) -> None:
        stats = {"clothing": "nackt", "mood": "", "hunger": "satt"}
        rendered = serialise_stats(stats)
        assert "clothing=nackt" in rendered
        assert "mood" in rendered and "mood=" not in rendered  # no value
        assert "hunger=satt" in rendered


# ─── Container lifecycle ────────────────────────────────────────────


@pytest.mark.asyncio
class TestRPSessionContainerLifecycle:
    async def test_create_yields_empty_namespace(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(
            tmp_path, "sess1", memory,
            title="Insel-A",
            scene="vier Mädchen am Strand",
            tracked_stats={"clothing": "nackt", "mood": ""},
        )
        try:
            assert ct.folder == tmp_path / "sess1"
            assert ct.folder.is_dir()
            assert ct.collection == "rp__sess1"
            # Chroma collection was registered with backend.
            assert "rp__sess1" in memory.ensured

            # Empty namespace contract.
            assert await ct.list_turns() == []
            assert await ct.list_messages() == []
            assert await ct.all_char_states() == {}
            assert await ct.get_char_state("any") == {}

            # Meta carries the seeded title/scene/stats.
            meta = await ct.get_meta()
            assert meta["title"] == "Insel-A"
            assert meta["scene"] == "vier Mädchen am Strand"
            assert meta["tracked_stats"] == {"clothing": "nackt", "mood": ""}
        finally:
            await ct.close()

    async def test_create_then_open_roundtrip(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        sid = "sess2"
        ct1 = await RPSessionContainer.create(tmp_path, sid, memory, title="X")
        await ct1.append_message({"role": "user", "content": "Hi"})
        await ct1.set_char_state("char_a", {"mood": "happy"})
        await ct1.close()

        ct2 = await RPSessionContainer.open(tmp_path, sid, memory)
        try:
            assert (await ct2.get_meta())["title"] == "X"
            assert await ct2.list_messages() == [{"role": "user", "content": "Hi"}]
            assert await ct2.get_char_state("char_a") == {"mood": "happy"}
        finally:
            await ct2.close()

    async def test_create_refuses_if_folder_exists(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        await RPSessionContainer.create(tmp_path, "dup", memory)
        with pytest.raises(FileExistsError):
            await RPSessionContainer.create(tmp_path, "dup", memory)

    async def test_open_refuses_if_folder_missing(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await RPSessionContainer.open(tmp_path, "ghost", memory)

    async def test_destroy_removes_folder_and_collection(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(tmp_path, "doomed", memory)
        await ct.memory_write(text="hi", character_id="c1")
        assert (tmp_path / "doomed").is_dir()
        assert "rp__doomed" in memory.items

        await ct.destroy()

        assert not (tmp_path / "doomed").exists()
        assert "rp__doomed" in memory.deleted
        assert "rp__doomed" not in memory.items


# ─── Turns ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRPSessionTurns:
    async def test_append_and_list(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(tmp_path, "t1", memory)
        try:
            await ct.append_turn(TurnRow(
                id="t-1", character_id="sandra", character_name="Sandra",
                round_id="r-1", order_num=0, content="*winkt*",
            ))
            await ct.append_turn(TurnRow(
                id="t-2", character_id="lena", character_name="Lena",
                round_id="r-1", order_num=1, content="*lacht*",
            ))
            turns = await ct.list_turns()
            assert [t.id for t in turns] == ["t-1", "t-2"]
            assert turns[0].character_name == "Sandra"
            assert turns[1].content == "*lacht*"
        finally:
            await ct.close()

    async def test_filter_by_character(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(tmp_path, "t2", memory)
        try:
            await ct.append_turn(TurnRow(
                id="a", character_id="sandra", character_name="Sandra",
                round_id="r1", order_num=0, content="A",
            ))
            await ct.append_turn(TurnRow(
                id="b", character_id="lena", character_name="Lena",
                round_id="r1", order_num=1, content="B",
            ))
            sandras = await ct.list_turns(character_id="sandra")
            assert [t.id for t in sandras] == ["a"]
        finally:
            await ct.close()

    async def test_update_and_delete(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(tmp_path, "t3", memory)
        try:
            await ct.append_turn(TurnRow(
                id="x", character_id="c", character_name="C",
                round_id="r", order_num=0, content="old", skipped=True,
            ))
            await ct.update_turn_content("x", "new content")
            t = await ct.get_turn("x")
            assert t is not None
            assert t.content == "new content"
            assert t.skipped is False  # update flips skipped off
            await ct.delete_turn("x")
            assert await ct.get_turn("x") is None
        finally:
            await ct.close()


# ─── Live state ─────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRPSessionState:
    async def test_set_get_roundtrip(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(
            tmp_path, "s1", memory,
            tracked_stats={"clothing": "", "mood": ""},
        )
        try:
            await ct.set_char_state("c1", {"clothing": "nackt", "mood": "ruhig"})
            assert await ct.get_char_state("c1") == {
                "clothing": "nackt", "mood": "ruhig",
            }
        finally:
            await ct.close()

    async def test_update_filters_to_tracked_stats(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        """Mike's invariant: only keys configured in the session
        survive — random LLM-emitted keys are dropped."""
        ct = await RPSessionContainer.create(
            tmp_path, "s2", memory,
            tracked_stats={"clothing": "", "mood": ""},
        )
        try:
            result = await ct.update_char_state("c1", {
                "clothing": "shirt",
                "mood": "happy",
                "horoscope": "fish",  # not tracked → dropped
            })
            assert result == {"clothing": "shirt", "mood": "happy"}
            assert "horoscope" not in result
        finally:
            await ct.close()

    async def test_update_with_empty_value_clears_key(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(
            tmp_path, "s3", memory,
            tracked_stats={"clothing": "", "mood": ""},
        )
        try:
            await ct.set_char_state("c1", {"clothing": "shirt", "mood": "happy"})
            result = await ct.update_char_state("c1", {"clothing": ""})
            assert result == {"mood": "happy"}
        finally:
            await ct.close()

    async def test_snapshot_template_seeds_from_defaults(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(
            tmp_path, "s4", memory,
            tracked_stats={"clothing": "nackt", "mood": "ruhig", "hunger": ""},
        )
        try:
            result = await ct.snapshot_template_for_char("c1")
            # Defaults with non-empty values get pre-populated.
            assert result == {"clothing": "nackt", "mood": "ruhig"}
            # Persisted.
            assert await ct.get_char_state("c1") == {
                "clothing": "nackt", "mood": "ruhig",
            }
        finally:
            await ct.close()

    async def test_snapshot_template_no_clobber_existing_state(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(
            tmp_path, "s5", memory,
            tracked_stats={"clothing": "nackt"},
        )
        try:
            await ct.set_char_state("c1", {"clothing": "shirt"})
            # Re-attach should NOT reset to defaults.
            result = await ct.snapshot_template_for_char("c1")
            assert result == {"clothing": "shirt"}
        finally:
            await ct.close()

    async def test_remove_char_state(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(
            tmp_path, "s6", memory,
            tracked_stats={"mood": ""},
        )
        try:
            await ct.set_char_state("c1", {"mood": "happy"})
            await ct.remove_char_state("c1")
            assert await ct.get_char_state("c1") == {}
        finally:
            await ct.close()


# ─── Memory ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRPSessionMemory:
    async def test_write_lands_in_session_collection(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(tmp_path, "m1", memory)
        try:
            await ct.memory_write(text="user heißt Mike", character_id="sandra")
            items = memory.items["rp__m1"]
            assert len(items) == 1
            assert items[0]["text"] == "user heißt Mike"
            assert items[0]["metadata"]["character_id"] == "sandra"
            assert items[0]["metadata"]["session_id"] == "m1"
            assert items[0]["metadata"]["source"] == "character_chat"
        finally:
            await ct.close()

    async def test_recall_filters_by_character(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        ct = await RPSessionContainer.create(tmp_path, "m2", memory)
        try:
            await ct.memory_write(text="user mag Erdbeeren", character_id="sandra")
            await ct.memory_write(text="user mag Bananen", character_id="lena")
            sandras = await ct.memory_recall(
                query="mag", character_id="sandra", limit=10,
            )
            assert len(sandras) == 1
            assert "Erdbeeren" in sandras[0]["text"]
        finally:
            await ct.close()

    async def test_two_containers_are_isolated(
        self, tmp_path: Path, memory: FakeMemory,
    ) -> None:
        """The Phase 13 invariant Mike paid us to enforce."""
        a = await RPSessionContainer.create(tmp_path, "isle_a", memory)
        b = await RPSessionContainer.create(tmp_path, "isle_b", memory)
        try:
            await a.memory_write(text="strand voller Muscheln", character_id="sandra")
            # Same character, different session — must be invisible.
            hits_in_b = await b.memory_recall(
                query="muscheln", character_id="sandra", limit=10,
            )
            assert hits_in_b == []
            # And A's own recall sees it.
            hits_in_a = await a.memory_recall(
                query="muscheln", character_id="sandra", limit=10,
            )
            assert len(hits_in_a) == 1
        finally:
            await a.close()
            await b.close()
