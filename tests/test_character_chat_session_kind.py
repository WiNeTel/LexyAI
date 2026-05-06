"""
Phase 9.12 — character_chat plugin auto-tags sessions as kind=rp on
the first character attach.

The mechanism is small but important: when a user attaches a character
to a previously-plain session, the plugin flips
``session_store.set_kind(session_id, "rp")``. The new RP-tab listens
for ``session_kind_changed`` so it can show the upgraded session.

Tests:

1. Attaching a character to a chat-kind session sets the kind to "rp"
   and broadcasts ``session_kind_changed``.
2. Attaching to a session that's already "rp" does nothing extra (no
   redundant broadcast — set_kind returned False).
3. If the store is None (test stub) we silently no-op.
4. If session_store is missing entirely (older runtime), we silently
   no-op rather than blowing up the attach.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.character_chat.character_chat_plugin import CharacterChatPlugin


class _FakeSessionStore:
    """Tracks set_kind calls so the test can assert on them."""

    def __init__(self, sessions: dict[str, str]) -> None:
        # session_id → current kind. Missing sessions return False from
        # set_kind, mirroring the real store.
        self._sessions = dict(sessions)
        self.calls: list[tuple[str, str]] = []

    def set_kind(self, session_id: str, kind: str) -> bool:
        self.calls.append((session_id, kind))
        if session_id not in self._sessions:
            return False
        if self._sessions[session_id] == kind:
            return False
        self._sessions[session_id] = kind
        return True


class _FakeApp:
    """Stand-in for the LexyApp the plugin pulls session_store off."""

    def __init__(self, session_store: Any) -> None:
        self.session_store = session_store


class _FakeAPI:
    def __init__(self, session_store: Any) -> None:
        self._app = _FakeApp(session_store)
        self.broadcasts: list[dict[str, Any]] = []

    async def ws_broadcast(self, payload: dict[str, Any]) -> None:
        self.broadcasts.append(dict(payload))


def _build_plugin(session_store: Any) -> CharacterChatPlugin:
    plugin = CharacterChatPlugin.__new__(CharacterChatPlugin)
    plugin.api = _FakeAPI(session_store)
    return plugin


@pytest.mark.asyncio
async def test_attach_flips_chat_session_to_rp() -> None:
    store = _FakeSessionStore({"s1": "chat"})
    plugin = _build_plugin(store)
    await plugin._maybe_tag_session_rp("s1")
    assert ("s1", "rp") in store.calls
    assert store._sessions["s1"] == "rp"
    # One broadcast — frontend uses this to re-route the session.
    assert plugin.api.broadcasts == [
        {"type": "session_kind_changed", "session_id": "s1", "kind": "rp"}
    ]


@pytest.mark.asyncio
async def test_attach_to_rp_session_does_not_rebroadcast() -> None:
    store = _FakeSessionStore({"s1": "rp"})
    plugin = _build_plugin(store)
    await plugin._maybe_tag_session_rp("s1")
    # set_kind was still *called* (idempotency check happens inside the
    # store), but it returned False so no broadcast went out.
    assert store.calls == [("s1", "rp")]
    assert plugin.api.broadcasts == []


@pytest.mark.asyncio
async def test_attach_to_unknown_session_silently_skips() -> None:
    """If the session isn't in the store yet (race), we don't broadcast.
    Real-life scenario: attach happens before register_empty completes."""
    store = _FakeSessionStore({})
    plugin = _build_plugin(store)
    await plugin._maybe_tag_session_rp("missing")
    assert plugin.api.broadcasts == []


@pytest.mark.asyncio
async def test_no_session_store_is_silent_noop() -> None:
    """Test stubs sometimes leave ``app.session_store=None``. We must
    not crash the attach flow in that case."""
    plugin = _build_plugin(session_store=None)
    await plugin._maybe_tag_session_rp("s1")
    assert plugin.api.broadcasts == []


@pytest.mark.asyncio
async def test_empty_session_id_is_silent_noop() -> None:
    """Some import paths pass an empty session_id when nobody's actively
    chatting yet. The auto-tag must not poke the store with ""."""
    store = _FakeSessionStore({"s1": "chat"})
    plugin = _build_plugin(store)
    await plugin._maybe_tag_session_rp("")
    assert store.calls == []
    assert plugin.api.broadcasts == []


@pytest.mark.asyncio
async def test_set_kind_value_error_is_swallowed() -> None:
    """If set_kind raises (e.g. someone passes a future kind), we
    swallow the error so the attach itself still succeeds."""

    class _RaisingStore:
        def set_kind(self, session_id: str, kind: str) -> bool:
            raise ValueError("nope")

    plugin = _build_plugin(_RaisingStore())
    await plugin._maybe_tag_session_rp("s1")  # must not raise
    assert plugin.api.broadcasts == []
