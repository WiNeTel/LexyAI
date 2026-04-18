"""
Tests for the memory wipe system:

* ``MemoryManager.wipe_collection`` — drops + recreates a single collection
* ``MemoryManager.wipe_all`` — clears every collection + FTS5
* ``DELETE /api/v1/memory/collection/{name}`` — HTTP path
* ``POST /api/v1/memory/wipe`` — the nuclear option with all three flags
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    loop.run_until_complete(app.shutdown())


def _seed(lexy_client: TestClient, collection: str = "facts") -> list[str]:
    """Store a few items and return their ids."""
    ids: list[str] = []
    for i, text in enumerate(["seed_item_one", "seed_item_two", "seed_item_three"]):
        resp = lexy_client.post(
            "/api/v1/memory/store",
            json={
                "text": f"{text} [{i}]",
                "collection": collection,
                "metadata": {"tag": "wipe-test"},
            },
        )
        assert resp.status_code == 200
        ids.append(resp.json()["id"])
    return ids


# ─── Single-collection wipe ────────────────────────────────────────────────


def test_wipe_collection_endpoint(lexy_client: TestClient) -> None:
    _seed(lexy_client, "facts")

    # Verify items exist first
    browse = lexy_client.get("/api/v1/memory/browse?collection=facts").json()
    assert browse["total"] > 0

    resp = lexy_client.delete("/api/v1/memory/collection/facts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "wiped"
    assert data["collection"] == "facts"
    assert data["chroma"] >= 3  # at least our seed items

    # Post-wipe the collection must be empty
    browse_after = lexy_client.get("/api/v1/memory/browse?collection=facts").json()
    assert browse_after["total"] == 0
    assert browse_after["items"] == []


def test_wipe_unknown_collection_404(lexy_client: TestClient) -> None:
    resp = lexy_client.delete("/api/v1/memory/collection/doesnotexist")
    assert resp.status_code == 404


# ─── Full wipe via POST ────────────────────────────────────────────────────


def test_wipe_requires_confirm_flag(lexy_client: TestClient) -> None:
    resp = lexy_client.post("/api/v1/memory/wipe", json={})
    assert resp.status_code == 400

    resp = lexy_client.post("/api/v1/memory/wipe", json={"confirm": False})
    assert resp.status_code == 400


def test_wipe_all_collections(lexy_client: TestClient) -> None:
    # Seed multiple collections
    _seed(lexy_client, "facts")
    _seed(lexy_client, "solutions")
    _seed(lexy_client, "context")

    resp = lexy_client.post(
        "/api/v1/memory/wipe",
        json={
            "confirm": True,
            "collections": True,
            "sessions": False,
            "plugin_data": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "wiped"
    assert "collections" in data
    assert data["collections"]["total_chroma"] >= 9  # 3 items * 3 collections
    assert data["collections"]["total_fts"] >= 0

    # Every collection must now be empty
    for col in ["facts", "solutions", "errors", "context"]:
        browse = lexy_client.get(f"/api/v1/memory/browse?collection={col}").json()
        assert browse["total"] == 0, f"{col} not empty after wipe_all"


def test_wipe_sessions_only(lexy_client: TestClient) -> None:
    app = lexy_client.app.state.lexy
    app.session_store.append_pair("wipe-test-1", "hi", "hello")
    app.session_store.append_pair("wipe-test-2", "moin", "moin")
    assert len(app.session_store.sessions()) >= 2

    resp = lexy_client.post(
        "/api/v1/memory/wipe",
        json={
            "confirm": True,
            "collections": False,
            "sessions": True,
            "plugin_data": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sessions"]["status"] == "cleared"
    assert data["sessions"]["dropped_sessions"] >= 2
    assert len(app.session_store.sessions()) == 0


def test_wipe_emits_event(lexy_client: TestClient) -> None:
    """core.memory_wiped event should be emitted so plugins can react."""
    app = lexy_client.app.state.lexy
    received: list[dict] = []

    async def handler(event):
        received.append(event.data)

    app.event_bus.on("core.memory_wiped", handler, source="test")
    try:
        resp = lexy_client.post(
            "/api/v1/memory/wipe",
            json={
                "confirm": True,
                "collections": False,
                "sessions": True,
                "plugin_data": False,
            },
        )
        assert resp.status_code == 200
        assert len(received) == 1
    finally:
        app.event_bus.off_all("test")


def test_wipe_all_flags_off_is_noop(lexy_client: TestClient) -> None:
    resp = lexy_client.post(
        "/api/v1/memory/wipe",
        json={
            "confirm": True,
            "collections": False,
            "sessions": False,
            "plugin_data": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "wiped"
    assert "collections" not in data
    assert "sessions" not in data
    assert "plugin_data" not in data


# ─── Store → wipe → store round-trip ───────────────────────────────────────


def test_can_store_after_wipe(lexy_client: TestClient) -> None:
    """After wiping, the collection must be usable again (not deleted
    entirely — MemoryManager recreates it)."""
    lexy_client.post(
        "/api/v1/memory/wipe",
        json={"confirm": True, "collections": True, "sessions": False},
    )

    resp = lexy_client.post(
        "/api/v1/memory/store",
        json={"text": "post_wipe_item", "collection": "facts", "metadata": {}},
    )
    assert resp.status_code == 200
    item_id = resp.json()["id"]
    assert item_id

    # And it should be retrievable
    browse = lexy_client.get("/api/v1/memory/browse?collection=facts").json()
    assert browse["total"] == 1
    assert any("post_wipe_item" in item.get("content", "") for item in browse["items"])


# ─── Plugin data wipe (opt-in, with isolated tmp path) ────────────────────


def test_plugin_data_wipe_respects_flag(lexy_client: TestClient, tmp_path: Path) -> None:
    """
    We don't actually delete the live plugins dir in the test (that would
    kill the shared fixture). Instead we check that with ``plugin_data:
    false`` the response doesn't mention a plugin_data section.
    """
    resp = lexy_client.post(
        "/api/v1/memory/wipe",
        json={"confirm": True, "collections": False, "sessions": False, "plugin_data": False},
    )
    assert resp.status_code == 200
    assert "plugin_data" not in resp.json()
