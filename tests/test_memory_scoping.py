"""
Tests for Phase-4 memory scoping.

Verifies that ``MemoryManager.store`` tags items with the right
``project_id`` (either explicit, resolved via session_store, or the
default fallback) and that ``recall`` / ``search_fts`` correctly filter
by project — including the "legacy item without project_id is still
visible" rule, which we need so pre-migration memory isn't lost.

We boot a real :class:`LexyApp` (same pattern as the other integration
tests in this suite) so ChromaDB + FTS + the embedding client are wired
up for real.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp
from lexy_core.memory.memory_manager import CROSS_PROJECT_SCOPE
from lexy_core.project import DEFAULT_PROJECT_ID


_TEST_TAG = "__pytest_memory_scoping__"


@pytest.fixture(scope="module")
def lexy_client() -> Iterator[TestClient]:
    """Boot a LexyApp once for all memory-scoping tests."""
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    client._lexy = app  # type: ignore[attr-defined]
    yield client
    # Wipe anything we stored under our tag; we don't have a cheap query
    # for "by tag" so we lean on the broad wipe of the ``context``
    # collection which is where we park all our test items.
    try:
        asyncio.get_event_loop().run_until_complete(
            app.memory.wipe_collection("context")
        )
    except Exception:  # noqa: BLE001
        pass
    # Drop test projects + sessions we created.
    for project in app.project_store.list(include_archived=True):
        if project.name.startswith(_TEST_TAG):
            app.project_store.delete(project.id)
    for sid in list(app.session_store.sessions()):
        if sid.startswith(_TEST_TAG):
            app.session_store.clear(sid)
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def _uniq(label: str) -> str:
    return f"{_TEST_TAG}{label}-{uuid.uuid4().hex[:6]}"


# ─── Store: project_id resolution ────────────────────────────────────────────


def test_store_defaults_to_default_project(lexy_client: TestClient) -> None:
    """Without session_id or explicit project_id → default project."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    unique_phrase = _uniq("defaultmem")
    item_id = asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{unique_phrase} is a unique token used only here",
            collection="context",
            metadata={"tag": _TEST_TAG},
        )
    )
    assert item_id

    # Recall scoped to the default project should see it.
    hits_default = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=unique_phrase,
            collection="context",
            project_id=DEFAULT_PROJECT_ID,
            limit=5,
        )
    )
    assert any(unique_phrase in h.get("content", "") for h in hits_default)


def test_store_uses_explicit_project_id(lexy_client: TestClient) -> None:
    """An explicit ``project_id`` in metadata is honoured verbatim."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    project = app.project_store.create(name=_uniq("explicit"))
    phrase = _uniq("explmem")

    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{phrase} belongs only to the new project",
            collection="context",
            metadata={"project_id": project.id, "tag": _TEST_TAG},
        )
    )

    # Visible under that project's scope
    hits_scoped = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=phrase,
            collection="context",
            project_id=project.id,
            limit=5,
        )
    )
    assert any(phrase in h.get("content", "") for h in hits_scoped)

    # NOT visible under an unrelated project's scope
    other = app.project_store.create(name=_uniq("other"))
    hits_other = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=phrase,
            collection="context",
            project_id=other.id,
            limit=5,
        )
    )
    assert all(phrase not in h.get("content", "") for h in hits_other)


def test_store_resolves_project_from_session_store(
    lexy_client: TestClient,
) -> None:
    """When only ``session_id`` is given, the manager looks up the
    session's project via the wired session store."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    project = app.project_store.create(name=_uniq("fromsession"))
    sid = _uniq("sess")
    app.session_store.register_empty(sid, project_id=project.id)

    phrase = _uniq("sessmem")
    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{phrase} via session resolution",
            collection="context",
            metadata={"session_id": sid, "tag": _TEST_TAG},
        )
    )

    hits = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=phrase,
            collection="context",
            project_id=project.id,
            limit=5,
        )
    )
    assert any(phrase in h.get("content", "") for h in hits)


# ─── Recall: filter correctness ──────────────────────────────────────────────


def test_recall_none_scope_sees_everything(lexy_client: TestClient) -> None:
    """``project_id=None`` → cross-project recall."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    a = app.project_store.create(name=_uniq("crossA"))
    b = app.project_store.create(name=_uniq("crossB"))

    phrase_a = _uniq("crossphrasea")
    phrase_b = _uniq("crossphraseb")

    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{phrase_a} content",
            collection="context",
            metadata={"project_id": a.id, "tag": _TEST_TAG},
        )
    )
    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{phrase_b} content",
            collection="context",
            metadata={"project_id": b.id, "tag": _TEST_TAG},
        )
    )

    hits_all = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query="content",
            collection="context",
            project_id=None,
            limit=20,
        )
    )
    contents = " | ".join(h.get("content", "") for h in hits_all)
    assert phrase_a in contents
    assert phrase_b in contents


def test_recall_cross_project_sentinel(lexy_client: TestClient) -> None:
    """The ``__all__`` sentinel behaves identically to ``project_id=None``."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    project = app.project_store.create(name=_uniq("sentinel"))
    phrase = _uniq("sentmem")
    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{phrase} content",
            collection="context",
            metadata={"project_id": project.id, "tag": _TEST_TAG},
        )
    )

    hits = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=phrase,
            collection="context",
            project_id=CROSS_PROJECT_SCOPE,
            limit=5,
        )
    )
    assert any(phrase in h.get("content", "") for h in hits)


def test_recall_isolation_between_projects(lexy_client: TestClient) -> None:
    """Two projects see only their own items when scoping is enabled."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    a = app.project_store.create(name=_uniq("isoA"))
    b = app.project_store.create(name=_uniq("isoB"))

    phrase_a = _uniq("isophrasea")
    phrase_b = _uniq("isophraseb")

    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=phrase_a,
            collection="context",
            metadata={"project_id": a.id, "tag": _TEST_TAG},
        )
    )
    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=phrase_b,
            collection="context",
            metadata={"project_id": b.id, "tag": _TEST_TAG},
        )
    )

    hits_a = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=phrase_a,
            collection="context",
            project_id=a.id,
            limit=5,
        )
    )
    contents_a = " | ".join(h.get("content", "") for h in hits_a)
    assert phrase_a in contents_a
    assert phrase_b not in contents_a

    hits_b = asyncio.get_event_loop().run_until_complete(
        app.memory.recall(
            query=phrase_b,
            collection="context",
            project_id=b.id,
            limit=5,
        )
    )
    contents_b = " | ".join(h.get("content", "") for h in hits_b)
    assert phrase_b in contents_b
    assert phrase_a not in contents_b


# ─── search_fts: pure BM25 scoping ───────────────────────────────────────────


def test_search_fts_respects_project_id(lexy_client: TestClient) -> None:
    """``search_fts(..., project_id=...)`` must honour the scope."""
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    scoped = app.project_store.create(name=_uniq("ftsScoped"))
    other = app.project_store.create(name=_uniq("ftsOther"))

    token = _uniq("ftstoken").replace("-", "")  # plain alphanumerics for FTS
    asyncio.get_event_loop().run_until_complete(
        app.memory.store(
            text=f"{token} story",
            collection="context",
            metadata={"project_id": scoped.id, "tag": _TEST_TAG},
        )
    )

    hits_scoped = asyncio.get_event_loop().run_until_complete(
        app.memory.search_fts(query=token, limit=10, project_id=scoped.id)
    )
    assert any(token in h.get("content", "") for h in hits_scoped)

    hits_other = asyncio.get_event_loop().run_until_complete(
        app.memory.search_fts(query=token, limit=10, project_id=other.id)
    )
    assert all(token not in h.get("content", "") for h in hits_other)


# ─── PluginAPI plumbing ──────────────────────────────────────────────────────


def test_plugin_api_memory_store_accepts_project_id(
    lexy_client: TestClient,
) -> None:
    """The facade must forward ``project_id`` into the manager."""
    from lexy_core.plugin_system.plugin_api import PluginAPI
    from lexy_core.plugin_system.plugin_manifest import PluginManifest

    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]
    manifest = PluginManifest(name="__pytest_memory_scoping__")
    api = PluginAPI(
        plugin_name="__pytest_memory_scoping__",
        app=app,
        manifest=manifest,
    )

    project = app.project_store.create(name=_uniq("apistore"))
    phrase = _uniq("apiphrase")

    item_id = asyncio.get_event_loop().run_until_complete(
        api.memory_store(
            text=f"{phrase} from plugin api",
            collection="context",
            project_id=project.id,
        )
    )
    assert item_id

    hits = asyncio.get_event_loop().run_until_complete(
        api.memory_recall(
            query=phrase,
            collection="context",
            project_id=project.id,
            limit=5,
        )
    )
    assert any(phrase in h.get("content", "") for h in hits)
