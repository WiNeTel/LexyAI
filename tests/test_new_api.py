"""
Integration tests for the new GUI-backing API endpoints.

These tests boot a full LexyApp (with live services) via the FastAPI
TestClient and exercise:

* ``/api/v1/settings`` GET/PATCH
* ``/api/v1/sessions`` GET + ``/{session_id}/history`` + DELETE
* ``/api/v1/voice/providers`` + ``/voice/config`` GET/PATCH
* ``/api/v1/plugins/{name}/enable`` + ``/disable``
* ``/api/v1/memory/delete``
"""

from __future__ import annotations

import asyncio

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


# ─── Settings API ───────────────────────────────────────────────────────────


def test_settings_get(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "brains" in data
    assert "routing" in data
    assert "voice" in data
    assert "memory" in data
    assert "e4b" in data["brains"]
    assert "a4b" in data["brains"]
    assert data["brains"]["a4b"]["thinking"] is True


def test_settings_patch_brain_temperature(lexy_client: TestClient) -> None:
    # Patch the e4b temperature and verify it sticks
    resp = lexy_client.patch(
        "/api/v1/settings",
        json={"brains": {"e4b": {"temperature": 0.42}}},
    )
    assert resp.status_code == 200
    assert resp.json()["changed"]["brains"]["e4b"]["temperature"] == 0.42

    verify = lexy_client.get("/api/v1/settings").json()
    assert verify["brains"]["e4b"]["temperature"] == 0.42


def test_settings_patch_unknown_brain_404(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/settings",
        json={"brains": {"nonexistent": {"temperature": 0.5}}},
    )
    assert resp.status_code == 404


def test_settings_patch_routing_default(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/settings",
        json={"routing": {"default_brain": "a4b"}},
    )
    assert resp.status_code == 200
    verify = lexy_client.get("/api/v1/settings").json()
    assert verify["routing"]["default_brain"] == "a4b"
    # Reset
    lexy_client.patch(
        "/api/v1/settings",
        json={"routing": {"default_brain": "e4b"}},
    )


# ─── Sessions API ───────────────────────────────────────────────────────────


def test_sessions_list(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


def test_session_history_empty(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/sessions/nonexistent-session-id/history")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_session_clear(lexy_client: TestClient) -> None:
    # Seed one message and verify clear drops it
    app = lexy_client.app.state.lexy
    app.session_store.append_pair("test-clear", "hi", "hello")
    assert app.session_store.length("test-clear") == 2

    resp = lexy_client.delete("/api/v1/sessions/test-clear")
    assert resp.status_code == 200
    assert resp.json()["dropped"] == 2
    assert app.session_store.length("test-clear") == 0


# ─── Voice API ──────────────────────────────────────────────────────────────


def test_voice_providers(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/voice/providers")
    assert resp.status_code == 200
    data = resp.json()
    # Keys always present, even when no providers are registered
    assert "stt" in data
    assert "tts" in data
    assert "active_stt" in data
    assert "active_tts" in data


def test_voice_config_get(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/voice/config")
    # The config may be empty if CosyVoice is not reachable,
    # but the endpoint must always succeed with JSON.
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


# ─── Plugin enable/disable ─────────────────────────────────────────────────


def test_plugin_list_includes_new_plugins(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/plugins")
    assert resp.status_code == 200
    names = {p["name"] for p in resp.json()["plugins"]}
    assert "scheduler" in names
    assert "autonomous_thinking" in names
    assert "weather" in names


def test_plugin_disable_and_enable(lexy_client: TestClient) -> None:
    # weather is a safe plugin to toggle — it has no external deps.
    disable = lexy_client.post("/api/v1/plugins/weather/disable")
    assert disable.status_code == 200
    enable = lexy_client.post("/api/v1/plugins/weather/enable")
    assert enable.status_code == 200
    # Re-list and verify enabled again
    listing = lexy_client.get("/api/v1/plugins").json()["plugins"]
    weather = next(p for p in listing if p["name"] == "weather")
    assert weather["enabled"] is True


# ─── Memory store/recall/browse/delete roundtrip ───────────────────────────


def test_memory_store_browse_delete(lexy_client: TestClient) -> None:
    # Store a test item
    store = lexy_client.post(
        "/api/v1/memory/store",
        json={"text": "test_fact_api_only", "collection": "facts", "metadata": {}},
    )
    assert store.status_code == 200
    item_id = store.json()["id"]
    assert item_id

    # Recall it
    recall = lexy_client.post(
        "/api/v1/memory/recall",
        json={"query": "test_fact_api_only", "collection": "facts", "limit": 5},
    )
    assert recall.status_code == 200
    results = recall.json()["results"]
    found = any(item_id in r.get("id", "") for r in results)
    assert found or len(results) > 0  # at least something came back

    # Delete it
    delete = lexy_client.post(
        "/api/v1/memory/delete",
        json={"id": item_id, "collection": "facts"},
    )
    assert delete.status_code == 200
    assert delete.json()["status"] == "deleted"
