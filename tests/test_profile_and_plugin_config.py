"""
Tests for:
* VRAM profile → active_brain_names / profile_excludes_plugin
* Plugin config GET/PATCH endpoints + persistence to plugins.yaml
* Self-signed SSL cert generator
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp
from lexy_core.config import (
    BrainConfig,
    LexyConfig,
    PluginsConfig,
    RoutingConfig,
    ServerConfig,
    SystemConfig,
)


# ─── VRAM profile helpers ───────────────────────────────────────────────────


def _config_with(profile: str) -> LexyConfig:
    cfg = LexyConfig(
        system=SystemConfig(profile=profile),
        server=ServerConfig(),
        brains={
            "e4b": BrainConfig(model="e4b", endpoint="http://localhost:5006/v1"),
            "a4b": BrainConfig(model="a4b", endpoint="http://localhost:5005/v1"),
            "multi": BrainConfig(model="multi", endpoint="http://localhost:5007/v1"),
        },
        routing=RoutingConfig(default_brain="a4b"),
        plugins=PluginsConfig(),
    )
    return cfg


def test_profile_chat_drops_only_multi_brain() -> None:
    """
    ``chat`` profile no longer excludes voice_gemma4 — with E4B running as
    a multimodal server on :5006, one process handles text + vision + STT
    and voice_gemma4 just points at that same endpoint.
    """
    cfg = _config_with("chat")
    assert cfg.active_brain_names() == {"e4b", "a4b"}
    assert cfg.profile_excludes_plugin("voice_gemma4") is False
    assert cfg.profile_excludes_plugin("voice_cosyvoice") is False
    assert cfg.profile_excludes_plugin("weather") is False


def test_profile_voice_drops_e4b() -> None:
    cfg = _config_with("voice")
    assert cfg.active_brain_names() == {"multi", "a4b"}
    assert cfg.profile_excludes_plugin("voice_gemma4") is False


def test_profile_full_keeps_everything() -> None:
    cfg = _config_with("full")
    assert cfg.active_brain_names() == {"e4b", "a4b", "multi"}
    assert cfg.profile_excludes_plugin("voice_gemma4") is False


def test_profile_unknown_falls_back_to_all() -> None:
    cfg = _config_with("experimental-mode")
    assert cfg.active_brain_names() == {"e4b", "a4b", "multi"}


# ─── SSL cert generator ─────────────────────────────────────────────────────


def test_ensure_cert_creates_files(tmp_path: Path) -> None:
    from lexy_core.utils.ssl_utils import ensure_cert

    cert = tmp_path / "test.crt"
    key = tmp_path / "test.key"
    assert not cert.exists()
    assert not key.exists()

    cert_path, key_path = ensure_cert(cert, key)
    assert cert_path.exists()
    assert key_path.exists()
    assert cert_path.read_bytes().startswith(b"-----BEGIN CERTIFICATE-----")
    assert b"-----BEGIN" in key_path.read_bytes()


def test_ensure_cert_idempotent(tmp_path: Path) -> None:
    from lexy_core.utils.ssl_utils import ensure_cert

    cert = tmp_path / "test.crt"
    key = tmp_path / "test.key"
    ensure_cert(cert, key)
    first_cert = cert.read_bytes()
    first_key = key.read_bytes()
    # Second call must not regenerate
    ensure_cert(cert, key)
    assert cert.read_bytes() == first_cert
    assert key.read_bytes() == first_key


# ─── Plugin config API ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    loop.run_until_complete(app.shutdown())


def test_get_plugin_config_weather(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/plugins/weather/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "weather"
    assert "defaults" in data
    assert "effective" in data
    # The weather plugin has a `location` default
    assert "location" in data["defaults"]


def test_get_plugin_config_unknown_404(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/plugins/nonexistent/config")
    assert resp.status_code == 404


def test_patch_plugin_config_persists(lexy_client: TestClient) -> None:
    # Patch the weather plugin's location
    resp = lexy_client.patch(
        "/api/v1/plugins/weather/config",
        json={"location": "TestTown", "units": "metric"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["restart_required"] is True
    assert data["overrides"]["location"] == "TestTown"

    # Verify the write actually hit plugins.yaml
    plugins_yaml = Path("config/plugins.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(plugins_yaml)
    assert "weather" in parsed
    assert parsed["weather"]["location"] == "TestTown"

    # Re-GET and verify overrides are reported
    verify = lexy_client.get("/api/v1/plugins/weather/config").json()
    assert verify["overrides"]["location"] == "TestTown"
    assert verify["effective"]["location"] == "TestTown"


def test_patch_plugin_config_unknown_404(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/plugins/nonexistent/config", json={"key": "value"}
    )
    assert resp.status_code == 404


# ─── Plugin status endpoint ─────────────────────────────────────────────


def test_get_plugin_status_for_autonomous_thinking(lexy_client: TestClient) -> None:
    """autonomous_thinking exposes ``get_status()`` → UI-friendly snapshot."""
    resp = lexy_client.get("/api/v1/plugins/autonomous_thinking/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "autonomous_thinking"
    assert data["available"] is True
    # Core observability fields the dashboard reads.
    for key in (
        "active",
        "loop_alive",
        "in_quiet_hours",
        "idle_seconds",
        "mode_interval_seconds",
        "last_tick_at",
        "last_skip_reason",
        "last_thought_at",
        "last_thought_mode",
        "thoughts_last_hour",
        "total_thoughts",
        "next_tick_in_seconds",
    ):
        assert key in data, f"status payload missing {key!r}"


def test_get_plugin_status_for_plugin_without_hook(
    lexy_client: TestClient,
) -> None:
    """Plugins that don't expose ``get_status()`` get a clean
    ``available: False`` response, never a 500."""
    # weather doesn't implement get_status.
    resp = lexy_client.get("/api/v1/plugins/weather/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "weather"
    assert data["available"] is False


def test_get_plugin_status_unknown_404(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/plugins/nonexistent/status")
    assert resp.status_code == 404


# ─── Settings system patch (profile switch) ────────────────────────────────


def test_settings_patch_profile(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/settings",
        json={"system": {"profile": "voice"}},
    )
    assert resp.status_code == 200
    assert resp.json()["changed"]["system"]["profile"] == "voice"
    verify = lexy_client.get("/api/v1/settings").json()
    assert verify["system"]["profile"] == "voice"
    # Reset so other tests aren't polluted
    lexy_client.patch("/api/v1/settings", json={"system": {"profile": "chat"}})
