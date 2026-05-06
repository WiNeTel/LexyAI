"""Smoke tests for the FastAPI gateway + static frontend mount."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    """Boot a LexyApp once for all gateway tests."""
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def test_root_redirects_to_static(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers.get("location", "").endswith("/static/index.html")


def test_index_html_served(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/static/index.html")
    assert resp.status_code == 200
    assert "Lexy AI" in resp.text
    assert 'id="chat-window"' in resp.text


def test_static_assets_served(lexy_client: TestClient) -> None:
    css = lexy_client.get("/static/style.css")
    js = lexy_client.get("/static/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert b"--accent" in css.content
    assert b"WebSocket" in js.content


def test_health_endpoint(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "services" in data
    assert "version" in data


def test_plugins_list_endpoint(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/plugins")
    assert resp.status_code == 200
    plugins = resp.json().get("plugins", [])
    names = {p["name"] for p in plugins}
    assert "weather" in names
    assert "voice_cosyvoice" in names


# ─── Phase 9.10 — plugin status + structured enable errors ───────────────


def test_plugin_status_unknown_returns_404(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/plugins/__no_such_plugin__/status")
    assert resp.status_code == 404


def test_plugin_status_returns_structured_payload(lexy_client: TestClient) -> None:
    """Phase 9.10: ``/status`` is the single source of truth that
    the dashboard and the plugin tab both read from.

    We pick ``voice_cosyvoice`` because that's the plugin the bug
    report was about; it now exposes ``last_error`` /
    ``module_importable`` / ``server_url`` via ``get_status()``."""
    resp = lexy_client.get("/api/v1/plugins/voice_cosyvoice/status")
    assert resp.status_code == 200
    body = resp.json()
    # Core keys every plugin gets from the gateway:
    assert body["name"] == "voice_cosyvoice"
    assert "loaded" in body
    assert "enabled" in body
    assert "version" in body
    # Custom keys merged from the plugin's get_status():
    if body.get("loaded"):
        # Plugin-specific contract — see plugins/voice_cosyvoice/cosyvoice_plugin.py
        assert "last_error" in body
        assert "server_url" in body


def test_enable_unknown_plugin_returns_structured_422(
    lexy_client: TestClient,
) -> None:
    """The old route raised ``HTTPException(500)`` on any failure
    which surfaced as ``HTTP 500`` in the toast. Phase 9.10 returns
    ``422`` with ``detail.code = "plugin_enable_failed"`` so the
    frontend can show the actual error instead of a status code."""
    resp = lexy_client.post("/api/v1/plugins/__no_such_plugin__/enable")
    assert resp.status_code == 422
    detail = resp.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("code") == "plugin_enable_failed"
    assert detail.get("plugin") == "__no_such_plugin__"
    assert "error" in detail
