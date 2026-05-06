"""Tests for the pulse-staleness skip (Phase 9.7).

Mike's audit: "sollten die Charakter Pulse nicht nur dann Aktiv sein,
wenn der charakter auch genutzt wird?"

Before this fix every pulse fired regardless of whether Mike had the
session open or even existed any more — six-hour-old chats kept
producing ghost replies nobody read. The fix consults
``SessionStore.get_meta(session_id).updated_at`` and skips firing when
the session has been idle for longer than
``pulse_session_stale_seconds``.

These tests pin down the staleness decision tree:

* Recent session → not stale → pulse fires.
* Idle session beyond threshold → stale → pulse skipped.
* Threshold 0 disables the check entirely (= old behaviour).
* Unknown / 0-timestamp session → stale (= safer default).
* Real ``SessionStore`` integration — ``register_empty`` makes a
  session non-stale, ``append_user`` keeps it non-stale, time-warping
  the meta clobbers it back to stale.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lexy_core.agent.session_store import SessionStore
from plugins.character_chat.character_chat_plugin import CharacterChatPlugin


# ─── Pure-function: _is_session_stale ────────────────────────────────


class _StubAPI:
    """Minimal PluginAPI stand-in — only the bits ``_is_session_stale``
    looks at. We build it bare-handed so the test doesn't have to boot
    a full ``LexyApp``."""

    def __init__(self, session_store: Any = None) -> None:
        # The plugin reaches the real session_store via
        # ``self.api._app.session_store``. We mirror that path.
        self._app = MagicMock()
        self._app.session_store = session_store


def _make_stub_plugin(
    session_store: Any = None,
    stale_seconds: float = 21600.0,
) -> CharacterChatPlugin:
    """Build a CharacterChatPlugin instance without invoking on_load.

    We hand-set the two attributes ``_is_session_stale`` reads
    (``api._app.session_store`` and ``_pulse_session_stale_seconds``)
    so the helper can run in isolation.
    """
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin.api = _StubAPI(session_store=session_store)
    plugin._pulse_session_stale_seconds = float(stale_seconds)
    return plugin


class TestIsSessionStaleHelper:
    def test_disabled_with_zero_threshold(self) -> None:
        store = SessionStore(max_messages=20)
        store.register_empty("s1")
        plugin = _make_stub_plugin(session_store=store, stale_seconds=0)
        # Threshold = 0 → the check is off, regardless of timestamps.
        assert plugin._is_session_stale("s1") is False
        assert plugin._is_session_stale("never-existed") is False

    def test_recent_session_not_stale(self) -> None:
        store = SessionStore(max_messages=20)
        store.register_empty("s1")
        plugin = _make_stub_plugin(session_store=store, stale_seconds=3600)
        assert plugin._is_session_stale("s1") is False

    def test_idle_session_is_stale(self) -> None:
        store = SessionStore(max_messages=20)
        store.register_empty("s1")
        # Time-warp the entry: pretend it was last touched 8 hours ago.
        # The store keeps meta in-memory, no need to hit disk for this.
        store._sessions["s1"]["meta"]["updated_at"] = time.time() - 8 * 3600
        plugin = _make_stub_plugin(session_store=store, stale_seconds=6 * 3600)
        assert plugin._is_session_stale("s1") is True

    def test_unknown_session_is_stale(self) -> None:
        store = SessionStore(max_messages=20)
        plugin = _make_stub_plugin(session_store=store, stale_seconds=3600)
        assert plugin._is_session_stale("never-existed") is True

    def test_zero_updated_at_is_stale(self) -> None:
        # Legacy v1-migrated session with no real timestamp → meta has
        # updated_at = 0.0. Treat as stale (= "we don't know if you've
        # touched this since the migration; better to skip than spam").
        store = SessionStore(max_messages=20)
        store._sessions["legacy"] = {
            "messages": [],
            "meta": {
                "project_id": "default",
                "created_at": 0.0,
                "updated_at": 0.0,
                "title": None,
            },
        }
        plugin = _make_stub_plugin(session_store=store, stale_seconds=3600)
        assert plugin._is_session_stale("legacy") is True

    def test_session_store_missing_returns_false(self) -> None:
        # Defensive: when the SessionStore isn't there at all (some test
        # setup), the helper falls back to "not stale" so we don't
        # accidentally mute every pulse during boot.
        plugin = _make_stub_plugin(session_store=None, stale_seconds=3600)
        assert plugin._is_session_stale("any") is False

    def test_threshold_boundary_just_inside(self) -> None:
        store = SessionStore(max_messages=20)
        store.register_empty("s1")
        # Just inside the window — 5 min ago, threshold 10 min.
        store._sessions["s1"]["meta"]["updated_at"] = time.time() - 5 * 60
        plugin = _make_stub_plugin(session_store=store, stale_seconds=10 * 60)
        assert plugin._is_session_stale("s1") is False

    def test_threshold_boundary_just_outside(self) -> None:
        store = SessionStore(max_messages=20)
        store.register_empty("s1")
        # Just outside — 11 min ago, threshold 10 min.
        store._sessions["s1"]["meta"]["updated_at"] = time.time() - 11 * 60
        plugin = _make_stub_plugin(session_store=store, stale_seconds=10 * 60)
        assert plugin._is_session_stale("s1") is True


# ─── Real SessionStore integration ──────────────────────────────────


class TestStalenessAgainstRealStore:
    """Boot a real SessionStore and verify the activity-tracking
    interactions produce the expected staleness flags."""

    def test_append_user_keeps_session_fresh(self, tmp_path: Path) -> None:
        store = SessionStore(
            max_messages=20, persistent_path=tmp_path / "sessions.json"
        )
        store.register_empty("s1")
        # Time-warp to make it stale.
        store._sessions["s1"]["meta"]["updated_at"] = time.time() - 9_000
        plugin = _make_stub_plugin(session_store=store, stale_seconds=3600)
        assert plugin._is_session_stale("s1") is True
        # User sends a message → ``_touch`` updates ``updated_at``.
        store.append_user("s1", "hi")
        assert plugin._is_session_stale("s1") is False

    def test_set_project_resets_staleness(self, tmp_path: Path) -> None:
        store = SessionStore(
            max_messages=20, persistent_path=tmp_path / "sessions.json"
        )
        store.register_empty("s1")
        store._sessions["s1"]["meta"]["updated_at"] = time.time() - 9_000
        plugin = _make_stub_plugin(session_store=store, stale_seconds=3600)
        assert plugin._is_session_stale("s1") is True
        # Any meta mutation calls ``_touch`` → updated_at = now.
        store.set_project("s1", "lexy")
        assert plugin._is_session_stale("s1") is False

    def test_pop_last_pair_does_not_keep_stale_alive(
        self, tmp_path: Path
    ) -> None:
        # Edge case: ``pop_last_pair`` calls ``_touch`` only when it
        # actually removed something. If the session is empty, it should
        # leave ``updated_at`` as-is — so a stale empty session stays
        # stale.
        store = SessionStore(
            max_messages=20, persistent_path=tmp_path / "sessions.json"
        )
        store.register_empty("empty-stale")
        store._sessions["empty-stale"]["meta"]["updated_at"] = time.time() - 9_000
        plugin = _make_stub_plugin(session_store=store, stale_seconds=3600)
        assert plugin._is_session_stale("empty-stale") is True
        u, a = store.pop_last_pair("empty-stale")
        assert u is None and a is None
        assert plugin._is_session_stale("empty-stale") is True
