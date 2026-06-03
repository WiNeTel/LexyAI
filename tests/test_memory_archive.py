"""
Tests for the recoverable-archive primitive on :class:`MemoryManager`
(Phase P0 of the Hermes-inspired self-cleaning upgrade).

Unlike the other memory integration tests, these run **hermetically**: an
in-memory ChromaDB (``EphemeralClient``) replaces the HTTP client and a
deterministic fake embedding replaces the Jina model, so the archive /
restore / purge logic and the "archives never appear in recall" invariant
are verified without any running services.

Covered invariants
-------------------
* archiving removes an item from recall (targeted *and* search-all) and its
  FTS row, while copying it into ``__archive__<collection>``;
* restore is an exact metadata round-trip and empties the archive entry;
* ``purge_archive`` only deletes items archived before the cutoff;
* archiving a disabled / unknown / archive-prefixed collection is a no-op.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Iterator

import chromadb
import pytest

from lexy_core.config import MemoryConfig
from lexy_core.memory.memory_manager import ARCHIVE_PREFIX, MemoryManager


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeEmbedding:
    """Deterministic, offline embedding: identical text → identical vector
    (L2 distance 0 → perfect self-recall), different text → different vector.
    """

    dim = 32

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for char in text:
            vec[ord(char) % self.dim] += 1.0
        if not any(vec):
            vec[0] = 1.0
        return vec


@pytest.fixture()
def make_manager(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., MemoryManager]]:
    """Factory that builds initialised, isolated MemoryManagers."""
    # Each manager gets its own on-disk ChromaDB under a unique tmp path so
    # tests are fully isolated. (EphemeralClient shares one in-memory system
    # per process, which would leak archived items across tests.)
    counter = {"n": 0}

    def _client_factory(host: Any, port: Any) -> Any:
        counter["n"] += 1
        return chromadb.PersistentClient(path=str(tmp_path / f"chroma_{counter['n']}"))

    monkeypatch.setattr(chromadb, "HttpClient", _client_factory)
    created: list[MemoryManager] = []

    def _make(**cfg_kwargs: Any) -> MemoryManager:
        cfg = MemoryConfig(
            fts_db_path=str(tmp_path / f"fts_{len(created)}.db"), **cfg_kwargs
        )
        manager = MemoryManager(cfg, _FakeEmbedding())
        _run(manager.initialize())
        created.append(manager)
        return manager

    yield _make

    for manager in created:
        _run(manager.shutdown())


@pytest.fixture()
def manager(make_manager: Callable[..., MemoryManager]) -> MemoryManager:
    return make_manager()


def _ids(hits: list[dict[str, Any]]) -> set[str]:
    return {hit["id"] for hit in hits}


# ─── Archive removes from recall + FTS, lands in archive ────────────────────


def test_archive_excludes_from_recall_and_lands_in_archive(
    manager: MemoryManager,
) -> None:
    phrase = "archivetoken alpha bravo charlie"
    item_id = _run(manager.store(phrase, collection="context"))
    assert item_id

    before = _run(manager.recall(phrase, collection="context", limit=5))
    assert item_id in _ids(before)

    result = _run(manager.archive_items("context", [item_id], reason="test"))
    assert result["archived"] == 1

    after = _run(manager.recall(phrase, collection="context", limit=5))
    assert item_id not in _ids(after)

    archived, total = _run(manager.browse_archive("context"))
    assert total >= 1
    assert item_id in {it["id"] for it in archived}
    # The archive sibling must carry the bookkeeping tags.
    entry = next(it for it in archived if it["id"] == item_id)
    assert entry["metadata"]["archive_reason"] == "test"
    assert entry["metadata"]["origin_collection"] == "context"


def test_archive_strips_fts_row(manager: MemoryManager) -> None:
    token = "ftsuniquetoken" + uuid.uuid4().hex[:6]
    item_id = _run(manager.store(f"{token} body text", collection="facts"))

    fts_before = _run(manager.search_fts(token, limit=10))
    assert item_id in _ids(fts_before)

    _run(manager.archive_items("facts", [item_id], reason="test"))

    fts_after = _run(manager.search_fts(token, limit=10))
    assert item_id not in _ids(fts_after)


def test_archive_excluded_from_search_all(manager: MemoryManager) -> None:
    phrase = "searchall token delta echo"
    item_id = _run(manager.store(phrase, collection="context"))
    _run(manager.archive_items("context", [item_id], reason="test"))

    # collection=None → "search all" must never sweep the archive.
    hits = _run(manager.recall(phrase, collection=None, limit=10))
    assert item_id not in _ids(hits)
    # And the archive collection stayed out of the searchable cache.
    assert ARCHIVE_PREFIX + "context" not in manager._collections  # noqa: SLF001


# ─── Restore is an exact round-trip ─────────────────────────────────────────


def test_restore_round_trip_is_exact(manager: MemoryManager) -> None:
    phrase = "restoreme token foxtrot golf"
    item_id = _run(
        manager.store(
            phrase,
            collection="context",
            metadata={"project_id": "proj-x", "custom": "keepme"},
        )
    )
    _run(
        manager.archive_items(
            "context", [item_id], reason="dedup", extra_meta={"merged_into": "xyz"}
        )
    )
    gone = _run(manager.recall(phrase, collection="context", limit=5))
    assert item_id not in _ids(gone)

    result = _run(manager.restore_items("context", [item_id]))
    assert result["restored"] == 1

    back = _run(manager.recall(phrase, collection="context", limit=5))
    match = [hit for hit in back if hit["id"] == item_id]
    assert match
    meta = match[0].get("metadata") or {}
    # Original fields preserved; archive bookkeeping + extra_meta stripped.
    assert meta.get("custom") == "keepme"
    assert meta.get("project_id") == "proj-x"
    for tag in ("archived_at", "archive_reason", "origin_id", "_origin_meta", "merged_into"):
        assert tag not in meta

    # Archive entry consumed by the restore.
    _, total = _run(manager.browse_archive("context"))
    assert total == 0


# ─── Purge respects the cutoff ──────────────────────────────────────────────


def test_purge_archive_respects_cutoff(manager: MemoryManager) -> None:
    item_id = _run(manager.store("purge me hotel india", collection="errors"))
    _run(manager.archive_items("errors", [item_id], reason="test"))

    # Cutoff in the past → nothing is old enough → no purge.
    purged_none = _run(manager.purge_archive("errors", archived_before=time.time() - 3600))
    assert purged_none == 0
    _, total_still = _run(manager.browse_archive("errors"))
    assert total_still == 1

    # Cutoff in the future → the just-archived item qualifies → purged.
    purged = _run(manager.purge_archive("errors", archived_before=time.time() + 3600))
    assert purged == 1
    _, total_after = _run(manager.browse_archive("errors"))
    assert total_after == 0


# ─── No-op guards ───────────────────────────────────────────────────────────


def test_archive_disabled_is_noop(
    make_manager: Callable[..., MemoryManager]
) -> None:
    manager = make_manager(archive_enabled=False)
    phrase = "disabled archive juliet kilo"
    item_id = _run(manager.store(phrase, collection="facts"))

    result = _run(manager.archive_items("facts", [item_id], reason="test"))
    assert result == {"archived": 0, "fts": 0}

    # Item is untouched and still recallable.
    hits = _run(manager.recall(phrase, collection="facts", limit=5))
    assert item_id in _ids(hits)


def test_archive_unknown_collection_is_noop(manager: MemoryManager) -> None:
    assert _run(manager.archive_items("nope", ["x"], reason="t")) == {
        "archived": 0,
        "fts": 0,
    }
    # Refuse to recurse into an archive collection.
    assert _run(
        manager.archive_items(ARCHIVE_PREFIX + "facts", ["x"], reason="t")
    ) == {"archived": 0, "fts": 0}


# ─── Usage-based access tracking (decay signal) ─────────────────────────────


def test_recall_bumps_access_count(manager: MemoryManager) -> None:
    phrase = "accesstoken mike november oscar"
    item_id = _run(manager.store(phrase, collection="facts"))

    # Before any recall the field is absent (treated as 0 by decay).
    rows_before = _run(manager.get_by_ids("facts", [item_id]))
    assert rows_before[0]["metadata"].get("access_count", 0) == 0

    _run(manager.recall(phrase, collection="facts", limit=5))
    # The bump is fire-and-forget — let the scheduled task drain.
    _run(asyncio.sleep(0.1))

    rows_after = _run(manager.get_by_ids("facts", [item_id]))
    meta = rows_after[0]["metadata"]
    assert meta.get("access_count", 0) >= 1
    assert "last_accessed" in meta


def test_recall_track_access_disabled_is_noop(
    make_manager: Callable[..., MemoryManager]
) -> None:
    manager = make_manager(track_access=False)
    phrase = "notracking papa quebec"
    item_id = _run(manager.store(phrase, collection="facts"))
    _run(manager.recall(phrase, collection="facts", limit=5))
    _run(asyncio.sleep(0.1))
    rows = _run(manager.get_by_ids("facts", [item_id]))
    assert rows[0]["metadata"].get("access_count", 0) == 0
