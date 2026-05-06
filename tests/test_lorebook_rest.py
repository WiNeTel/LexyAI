"""
Phase 9.11 — REST endpoints for the Lorebook editor.

Mike asked for a UI to edit Lorebooks from inside Lexy. The Phase 9.8
backend already had ``LorebookStore`` + 8 WS handlers, but the editor
needs synchronous request/response — REST is cleaner than pairing WS
acks with originating UI actions. We added eight routes under
``/api/v1/plugins/character_chat/lorebooks*`` that delegate to the
plugin's ``_tool_lorebook_*`` and ``_tool_lore_entry_*`` methods.

These tests boot a real :class:`LexyApp`, then drive the routes via
the FastAPI :class:`TestClient`. Together they cover the full happy
path (create → list → patch → delete) plus the error cases the UI
relies on for branching:

* unknown lorebook id → 404 (lets the editor toast a clear message)
* invalid scope → 400 (so we don't smuggle bad scopes into the DB)
* entry without keys *or* always_on → 400 (entry would never fire)

We also test the ``scope`` filter in the list route — the editor's
left-pane scope dropdown depends on it.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


# ─── Lorebook CRUD ───────────────────────────────────────────────────


def _create_book(client: TestClient, **overrides) -> dict:
    payload = {"name": "Testwelt", "description": "for tests"}
    payload.update(overrides)
    resp = client.post(
        "/api/v1/plugins/character_chat/lorebooks", json=payload
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["book"]


def _delete_book(client: TestClient, book_id: str) -> None:
    """Best-effort cleanup — used in finally-blocks."""
    client.delete(
        f"/api/v1/plugins/character_chat/lorebooks/{book_id}"
    )


def test_create_lorebook_global(lexy_client: TestClient) -> None:
    book = _create_book(lexy_client, name="Test Phase 9.11 global")
    try:
        assert book["scope"] == "global"
        assert book["scope_id"] == ""
        assert book["enabled"] is True
        assert "id" in book and book["id"]
    finally:
        _delete_book(lexy_client, book["id"])


def test_create_character_scope_requires_scope_id(
    lexy_client: TestClient,
) -> None:
    """The store rejects character-scoped books without a scope_id; we
    surface that as 400 so the editor can highlight the field."""
    resp = lexy_client.post(
        "/api/v1/plugins/character_chat/lorebooks",
        json={"name": "missing-scope-id", "scope": "character"},
    )
    assert resp.status_code == 400


def test_create_invalid_scope_returns_400(
    lexy_client: TestClient,
) -> None:
    resp = lexy_client.post(
        "/api/v1/plugins/character_chat/lorebooks",
        json={"name": "bad-scope", "scope": "galactic"},
    )
    assert resp.status_code == 400


def test_list_lorebooks(lexy_client: TestClient) -> None:
    book = _create_book(lexy_client, name="Test Phase 9.11 list")
    try:
        resp = lexy_client.get("/api/v1/plugins/character_chat/lorebooks")
        assert resp.status_code == 200
        names = {b["name"] for b in resp.json()["books"]}
        assert "Test Phase 9.11 list" in names
    finally:
        _delete_book(lexy_client, book["id"])


def test_list_lorebooks_filter_scope(lexy_client: TestClient) -> None:
    """Filtering by ``scope=global`` must not return character-scoped
    or session-scoped books — the editor's left-pane scope dropdown
    relies on this filter being honoured."""
    book = _create_book(lexy_client, name="Test scope filter")
    try:
        resp = lexy_client.get(
            "/api/v1/plugins/character_chat/lorebooks?scope=global"
        )
        assert resp.status_code == 200
        for b in resp.json()["books"]:
            assert b["scope"] == "global"
    finally:
        _delete_book(lexy_client, book["id"])


def test_patch_lorebook(lexy_client: TestClient) -> None:
    book = _create_book(lexy_client, name="Test patch", token_budget=500)
    try:
        resp = lexy_client.patch(
            f"/api/v1/plugins/character_chat/lorebooks/{book['id']}",
            json={"name": "Renamed", "token_budget": 2222, "enabled": False},
        )
        assert resp.status_code == 200, resp.text
        updated = resp.json()["book"]
        assert updated["name"] == "Renamed"
        assert updated["token_budget"] == 2222
        assert updated["enabled"] is False
    finally:
        _delete_book(lexy_client, book["id"])


def test_patch_lorebook_unknown_returns_404(
    lexy_client: TestClient,
) -> None:
    resp = lexy_client.patch(
        "/api/v1/plugins/character_chat/lorebooks/__nope__",
        json={"name": "x"},
    )
    assert resp.status_code == 404


def test_delete_lorebook(lexy_client: TestClient) -> None:
    book = _create_book(lexy_client, name="Test delete")
    resp = lexy_client.delete(
        f"/api/v1/plugins/character_chat/lorebooks/{book['id']}"
    )
    assert resp.status_code == 200
    # Deleting twice → 404
    resp2 = lexy_client.delete(
        f"/api/v1/plugins/character_chat/lorebooks/{book['id']}"
    )
    assert resp2.status_code == 404


# ─── Entry CRUD ──────────────────────────────────────────────────────


def test_entry_create_list_patch_delete(lexy_client: TestClient) -> None:
    book = _create_book(lexy_client, name="Test entries CRUD")
    try:
        # Create
        create = lexy_client.post(
            f"/api/v1/plugins/character_chat/lorebooks/{book['id']}/entries",
            json={
                "name": "Hobbit-Trigger",
                "keys": ["Hobbit", "Auenland"],
                "content": "Hobbits sind kleine Halblinge.",
                "position": "before_scenario",
                "priority": 50,
            },
        )
        assert create.status_code == 200, create.text
        entry = create.json()["entry"]
        assert entry["keys"] == ["Hobbit", "Auenland"]
        assert entry["priority"] == 50

        # List
        listed = lexy_client.get(
            f"/api/v1/plugins/character_chat/lorebooks/{book['id']}/entries"
        )
        assert listed.status_code == 200
        ids = {e["id"] for e in listed.json()["entries"]}
        assert entry["id"] in ids

        # Patch
        patched = lexy_client.patch(
            f"/api/v1/plugins/character_chat/lore_entries/{entry['id']}",
            json={"content": "Hobbits leben im Auenland.", "priority": 5},
        )
        assert patched.status_code == 200
        updated = patched.json()["entry"]
        assert updated["priority"] == 5
        assert "Auenland" in updated["content"]

        # Delete
        deleted = lexy_client.delete(
            f"/api/v1/plugins/character_chat/lore_entries/{entry['id']}"
        )
        assert deleted.status_code == 200
        # Second delete → 404
        gone = lexy_client.delete(
            f"/api/v1/plugins/character_chat/lore_entries/{entry['id']}"
        )
        assert gone.status_code == 404
    finally:
        _delete_book(lexy_client, book["id"])


def test_entry_without_trigger_returns_400(lexy_client: TestClient) -> None:
    """An entry with no keys *and* always_on=False would never fire —
    the store rejects it, we surface 400 so the modal can highlight."""
    book = _create_book(lexy_client, name="Test entry no trigger")
    try:
        resp = lexy_client.post(
            f"/api/v1/plugins/character_chat/lorebooks/{book['id']}/entries",
            json={
                "name": "broken",
                "keys": [],
                "always_on": False,
                "content": "won't fire",
            },
        )
        assert resp.status_code == 400
    finally:
        _delete_book(lexy_client, book["id"])


def test_entry_always_on_without_keys_is_ok(lexy_client: TestClient) -> None:
    """The flip-side: always_on covers the trigger requirement, so a
    keyless always-on entry must succeed."""
    book = _create_book(lexy_client, name="Test always-on entry")
    try:
        resp = lexy_client.post(
            f"/api/v1/plugins/character_chat/lorebooks/{book['id']}/entries",
            json={
                "name": "world prologue",
                "keys": [],
                "always_on": True,
                "content": "Es war einmal…",
            },
        )
        assert resp.status_code == 200
        entry = resp.json()["entry"]
        assert entry["always_on"] is True
        assert entry["keys"] == []
    finally:
        _delete_book(lexy_client, book["id"])


def test_entry_invalid_position_returns_400(lexy_client: TestClient) -> None:
    book = _create_book(lexy_client, name="Test bad position")
    try:
        resp = lexy_client.post(
            f"/api/v1/plugins/character_chat/lorebooks/{book['id']}/entries",
            json={
                "name": "x",
                "keys": ["x"],
                "position": "into_the_void",
            },
        )
        assert resp.status_code == 400
    finally:
        _delete_book(lexy_client, book["id"])


def test_patch_entry_unknown_returns_404(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/plugins/character_chat/lore_entries/__nope__",
        json={"name": "x"},
    )
    assert resp.status_code == 404
