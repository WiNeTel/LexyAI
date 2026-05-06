"""
Phase 9.10 — CosyVoice plugin defensive-load tests.

Mike clicked the plugin's enable button and got a bare ``HTTP 500``.
The dashboard showed a green dot (= remote TTS server reachable) but
the plugin tab said "offline" (= local module wasn't importable on
this machine). Two separate failure modes were getting collapsed into
one opaque error.

The fix has three pieces, all covered here:

1. The plugin loads even when the local ``cosyvoice_tts`` module is
   missing or its constructor explodes — :meth:`on_load` swallows the
   error and stashes it in ``_last_error`` so the UI can surface it.
2. :meth:`get_status` returns a structured dict the dashboard and
   the plugin tab now both read from, so they stay consistent.
3. The two WS handlers respond with ``available=False`` + the
   stashed error instead of crashing when the provider isn't ready.

We don't try to import the real :mod:`cosyvoice_tts` here (it pulls
in optional deps that aren't installed on CI). Instead we patch the
relative import so each test exercises exactly one branch.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from plugins.voice_cosyvoice.cosyvoice_plugin import CosyVoicePlugin


class _FakeAPI:
    """Minimal :class:`PluginAPI` stand-in for cosyvoice tests."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self.registered_provider: tuple[str, Any] | None = None
        self.ws_handlers: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def register_voice_provider(self, kind: str, provider: Any) -> None:
        self.registered_provider = (kind, provider)

    def register_ws_handler(self, msg_type: str, handler: Any) -> None:
        self.ws_handlers[msg_type] = handler


def _make_plugin(config: dict[str, Any] | None = None) -> CosyVoicePlugin:
    api = _FakeAPI(config)
    manifest = MagicMock()
    manifest.config_defaults = {}
    return CosyVoicePlugin(api=api, manifest=manifest)


# ─── Module-import failure ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_load_survives_missing_module(monkeypatch):
    """The plugin must remain loadable when ``cosyvoice_tts`` is missing.

    Mike's machine had the package on ``sys.path`` but not the plugin's
    own dir, so the old top-level import blew up the entire enable
    flow with a 500. After the fix we want ``on_load`` to log + park
    the error and return cleanly.
    """
    plugin = _make_plugin()

    # Force the relative import inside on_load to fail with ImportError.
    real_import = __import__

    def _failing_import(name, *args, **kwargs):
        if name.endswith("cosyvoice_tts") or name == "plugins.voice_cosyvoice.cosyvoice_tts":
            raise ImportError("no module named cosyvoice_tts (test stub)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)

    await plugin.on_load()

    assert plugin._tts is None
    assert "module not importable" in plugin._last_error
    status = plugin.get_status()
    assert status["tts_provider_active"] is False
    assert status["module_importable"] is False
    assert status["last_error"]


# ─── Server-unreachable case ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_load_survives_unreachable_server(monkeypatch):
    """Module imports fine, ctor succeeds, but ``initialize()``
    returns False (server unreachable). Plugin must still load — only
    the provider stays unregistered and ``last_error`` explains why."""

    plugin = _make_plugin({"endpoint": "http://nope.invalid:5500"})

    fake_module = types.ModuleType("plugins.voice_cosyvoice.cosyvoice_tts")

    class _FakeTTS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize(self):
            return False

    fake_module.CosyVoiceTTS = _FakeTTS
    monkeypatch.setitem(
        sys.modules, "plugins.voice_cosyvoice.cosyvoice_tts", fake_module
    )

    await plugin.on_load()

    assert plugin._tts is None
    assert "server unreachable" in plugin._last_error
    status = plugin.get_status()
    assert status["module_importable"] is True  # we got past the import
    assert status["tts_provider_active"] is False
    assert "nope.invalid" in status["server_url"]


# ─── Happy path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_load_happy_path_registers_provider(monkeypatch):
    plugin = _make_plugin({"endpoint": "http://ok.example:5500"})

    fake_module = types.ModuleType("plugins.voice_cosyvoice.cosyvoice_tts")

    class _FakeTTS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def initialize(self):
            return True

        def get_config(self):
            return {"voice": "referenz_mio"}

        def update_config(self, cfg):
            self.kwargs.update(cfg)

        async def shutdown(self):
            pass

    fake_module.CosyVoiceTTS = _FakeTTS
    monkeypatch.setitem(
        sys.modules, "plugins.voice_cosyvoice.cosyvoice_tts", fake_module
    )

    await plugin.on_load()
    assert plugin._tts is not None
    assert plugin._last_error == ""
    assert plugin._initialized_at > 0

    await plugin.on_enable()
    assert plugin.api.registered_provider == ("tts", plugin._tts)
    assert "get_tts_config" in plugin.api.ws_handlers
    assert "update_tts_config" in plugin.api.ws_handlers


# ─── Degraded enable ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_enable_degraded_does_not_register_provider(monkeypatch):
    """When the TTS failed to initialise we still register the WS
    handlers so the UI can call ``get_tts_config`` and learn *why*
    we're degraded — but no provider gets registered."""
    plugin = _make_plugin()
    plugin._tts = None
    plugin._last_error = "server unreachable at http://nope:5500"

    await plugin.on_enable()
    assert plugin.api.registered_provider is None
    assert "get_tts_config" in plugin.api.ws_handlers
    assert "update_tts_config" in plugin.api.ws_handlers


# ─── WS handlers under degraded ───────────────────────────────────────────


class _RecordingClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_get_tts_config_handler_when_degraded():
    plugin = _make_plugin()
    plugin._tts = None
    plugin._last_error = "module not importable: foo"
    client = _RecordingClient()

    await plugin._handle_get_tts_config(client, {})
    assert client.sent
    msg = client.sent[0]
    assert msg["type"] == "tts_config"
    assert msg["available"] is False
    assert msg["last_error"] == "module not importable: foo"
    assert msg["config"] == {}


@pytest.mark.asyncio
async def test_update_tts_config_handler_when_degraded_returns_error():
    plugin = _make_plugin()
    plugin._tts = None
    plugin._last_error = "server unreachable"
    client = _RecordingClient()

    await plugin._handle_update_tts_config(client, {"config": {"voice": "x"}})
    assert client.sent
    msg = client.sent[0]
    assert msg["type"] == "error"
    assert "server unreachable" in msg["error"]


# ─── on_disable safety ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_disable_swallows_shutdown_failures():
    plugin = _make_plugin()

    class _BoomTTS:
        async def shutdown(self):
            raise RuntimeError("kaboom")

    plugin._tts = _BoomTTS()
    await plugin.on_disable()  # must not raise
    assert plugin._tts is None


@pytest.mark.asyncio
async def test_on_disable_when_no_tts_is_noop():
    plugin = _make_plugin()
    plugin._tts = None
    await plugin.on_disable()
    assert plugin._tts is None
