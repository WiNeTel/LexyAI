"""
Tests for Phase 8 — live plugin-config patches via the gateway.

The PATCH ``/api/v1/plugins/{name}/config`` endpoint now:
* persists the override to ``config/plugins.yaml`` (existing behaviour), AND
* calls ``on_config_changed(cfg)`` on the loaded plugin if it overrides it,
  so settings changes take effect without a restart.

We unit-test the hook-dispatch logic with a minimal fake ``LexyApp`` and
``FastAPI.TestClient`` — no ChromaDB / real plugins / disk writes beyond the
yaml override.

These tests deliberately do NOT share state with
``test_profile_and_plugin_config.py`` — they use their own tmp dir so the
real ``config/plugins.yaml`` is untouched.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lexy_core.plugin_system.base_plugin import BasePlugin
from lexy_core.websocket.gateway import build_app


# ─── Test plugins ──────────────────────────────────────────────────────


class _LivePlugin(BasePlugin):
    """Overrides on_config_changed — should be reported as applied_live."""

    def __init__(self) -> None:
        # Skip BasePlugin.__init__ to avoid needing real api/manifest objects.
        self.name = "live_plugin"
        self.calls: list[dict[str, Any]] = []

    async def on_load(self) -> None:  # pragma: no cover
        pass

    async def on_enable(self) -> None:  # pragma: no cover
        pass

    async def on_disable(self) -> None:  # pragma: no cover
        pass

    async def on_config_changed(self, cfg: dict[str, Any]) -> None:
        self.calls.append(dict(cfg))


class _StaticPlugin(BasePlugin):
    """Does NOT override on_config_changed — requires a restart."""

    def __init__(self) -> None:
        self.name = "static_plugin"

    async def on_load(self) -> None:  # pragma: no cover
        pass

    async def on_enable(self) -> None:  # pragma: no cover
        pass

    async def on_disable(self) -> None:  # pragma: no cover
        pass


# ─── Fake app surface ──────────────────────────────────────────────────


def _build_test_app(tmp_path: Path, plugins: dict[str, Any]) -> tuple[FastAPI, Any]:
    """Build a minimal LexyApp-shaped SimpleNamespace + FastAPI."""
    manifests = {
        name: SimpleNamespace(
            name=name,
            version="1.0.0",
            description=f"Test plugin {name}",
            config_defaults={"foo": "default"},
        )
        for name in plugins
    }

    plugin_loader = SimpleNamespace(
        get_plugin=lambda n: plugins.get(n),
        get_manifests=lambda: manifests,
    )

    # FastAPI-lifecycle bits the gateway expects
    app_state = SimpleNamespace(
        plugin_loader=plugin_loader,
        plugin_overrides={},
        config=SimpleNamespace(
            system=SimpleNamespace(name="Lexy Test", version="0.0.0"),
        ),
        llm=None,
        memory=None,
        voice=None,
        ws_server=None,
        event_bus=SimpleNamespace(emit=_async_noop),
        hooks=SimpleNamespace(),
        signals=SimpleNamespace(
            snapshot=lambda: {},
            get=lambda _key: SimpleNamespace(value="ok"),
        ),
        agent=None,
        session_store=None,
        project_store=None,
        channel_router=None,
        tool_registry=None,
        tool_caller=None,
        persona=None,
    )

    fastapi = build_app(app_state)  # type: ignore[arg-type]
    return fastapi, app_state


async def _async_noop(*args: Any, **kwargs: Any) -> None:
    return None


# ─── Tests ─────────────────────────────────────────────────────────────


def test_patch_plugin_that_overrides_hook_reports_applied_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _LivePlugin()
    fastapi, state = _build_test_app(tmp_path, {"live_plugin": live})

    # Redirect plugins.yaml writes into the tmp dir.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    client = TestClient(fastapi)
    resp = client.patch(
        "/api/v1/plugins/live_plugin/config",
        json={"foo": "bar", "tools_enabled": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["applied_live"] is True
    assert data["restart_required"] is False
    assert data["overrides"]["foo"] == "bar"

    # Plugin's on_config_changed was invoked with the merged config.
    assert len(live.calls) == 1
    merged = live.calls[0]
    assert merged["foo"] == "bar"
    assert merged["tools_enabled"] is False


def test_patch_plugin_without_override_still_persists_restart_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    static = _StaticPlugin()
    fastapi, state = _build_test_app(tmp_path, {"static_plugin": static})

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    client = TestClient(fastapi)
    resp = client.patch(
        "/api/v1/plugins/static_plugin/config",
        json={"foo": "bar"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied_live"] is False
    assert data["restart_required"] is True
    # Overrides are still in memory.
    assert state.plugin_overrides["static_plugin"]["foo"] == "bar"
    # And were written to disk.
    written = (tmp_path / "config" / "plugins.yaml").read_text(encoding="utf-8")
    assert "static_plugin" in written
    assert "bar" in written


def test_patch_plugin_hook_failure_falls_back_to_restart_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenPlugin(_LivePlugin):
        async def on_config_changed(self, cfg: dict[str, Any]) -> None:
            raise RuntimeError("bad config")

    broken = _BrokenPlugin()
    fastapi, state = _build_test_app(tmp_path, {"live_plugin": broken})

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    client = TestClient(fastapi)
    resp = client.patch(
        "/api/v1/plugins/live_plugin/config",
        json={"foo": "bar"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Hook raised → we report restart_required so the user knows to reload.
    assert data["applied_live"] is False
    assert data["restart_required"] is True


def test_patch_plugin_unknown_returns_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fastapi, _ = _build_test_app(tmp_path, {})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    client = TestClient(fastapi)
    resp = client.patch(
        "/api/v1/plugins/does_not_exist/config",
        json={"foo": "bar"},
    )
    assert resp.status_code == 404
