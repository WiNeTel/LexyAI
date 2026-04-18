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
