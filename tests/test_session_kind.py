"""
Phase 9.12 — Tests for the session ``kind`` field (Chat / Roleplay split).

The Chat-tab and the new Rollenspiel-tab both list sessions, but each
should only see *its own kind*. The mechanism is a simple
``meta.kind: "chat"|"rp"`` field on the session, plus a ``?kind=`` query
filter on ``GET /api/v1/sessions``.

These tests cover three layers:

1. **Store** (``SessionStore``) — default kind is ``"chat"``,
   ``set_kind`` flips it idempotently, unknown kinds raise, the field
   round-trips through save/load, legacy sessions without ``kind`` are
   backfilled to ``"chat"`` on load.
2. **Gateway** — ``GET /api/v1/sessions?kind=rp`` filters,
   ``GET /api/v1/sessions/{id}/history`` returns ``kind``,
   ``PATCH /api/v1/sessions/{id}`` accepts ``{kind}``.
3. **Backfill round-trip** — a v2 file written before 9.12 (no ``kind``
   key in meta) loads cleanly, gets defaulted to ``"chat"``, and is
   rewritten on disk with the field present.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lexy_core.agent.session_store import (
    SessionStore,
    VALID_SESSION_KINDS,
)
from lexy_core.app import LexyApp


# ─── Store-level tests ────────────────────────────────────────────────


def _store_with_path(tmpdir: Path) -> SessionStore:
    return SessionStore(
        max_messages=20,
        persistent_path=str(tmpdir / "sessions.json"),
    )


def test_new_session_defaults_to_chat_kind() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store_with_path(Path(tmp))
        store.register_empty("s1", project_id="p1")
        meta = store.get_meta("s1")
        assert meta["kind"] == "chat"


def test_set_kind_flips_to_rp() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store_with_path(Path(tmp))
        store.register_empty("s1")
        assert store.set_kind("s1", "rp") is True
        assert store.get_meta("s1")["kind"] == "rp"
        # Idempotent: same value → False (so callers skip needless broadcasts).
        assert store.set_kind("s1", "rp") is False


def test_set_kind_unknown_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store_with_path(Path(tmp))
        store.register_empty("s1")
        with pytest.raises(ValueError):
            store.set_kind("s1", "galactic")
        # Sanity: kind unchanged after the failed call.
        assert store.get_meta("s1")["kind"] == "chat"


def test_set_kind_unknown_session_returns_false() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store_with_path(Path(tmp))
        # No session registered — set_kind returns False without raising.
        assert store.set_kind("nope", "rp") is False


def test_kind_persists_across_save_load() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sessions.json"
        store1 = SessionStore(persistent_path=str(path))
        store1.register_empty("rp_session")
        store1.set_kind("rp_session", "rp")
        # Re-load from the same file — kind must come back as "rp".
        store2 = SessionStore(persistent_path=str(path))
        assert store2.get_meta("rp_session")["kind"] == "rp"


def test_legacy_session_without_kind_backfills_to_chat() -> None:
    """A session.json written before 9.12 has no ``kind`` key. Loading
    it must default each entry to ``"chat"`` (so the chat tab still
    sees them) and rewrite the file with the field present."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sessions.json"
        # Hand-craft a pre-9.12 v2 file.
        legacy = {
            "version": 2,
            "saved_at": 1700000000.0,
            "max_messages": 20,
            "sessions": {
                "old_session": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "meta": {
                        "project_id": "default",
                        "created_at": 1700000000.0,
                        "updated_at": 1700000000.0,
                        "title": "hi",
                        # no "kind"
                    },
                },
            },
        }
        path.write_text(json.dumps(legacy), encoding="utf-8")
        store = SessionStore(persistent_path=str(path))
        meta = store.get_meta("old_session")
        assert meta["kind"] == "chat"
        # File should have been rewritten with the new field.
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["sessions"]["old_session"]["meta"]["kind"] == "chat"


def test_invalid_kind_in_file_falls_back_to_chat() -> None:
    """If somebody hand-edits the file with a bogus kind ('galactic'),
    we must not crash on load — we coerce to the default."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sessions.json"
        bogus = {
            "version": 2,
            "saved_at": 1700000000.0,
            "max_messages": 20,
            "sessions": {
                "weird": {
                    "messages": [{"role": "user", "content": "hi"}],
                    "meta": {
                        "project_id": "default",
                        "created_at": 1700000000.0,
                        "updated_at": 1700000000.0,
                        "title": "hi",
                        "kind": "galactic",
                    },
                },
            },
        }
        path.write_text(json.dumps(bogus), encoding="utf-8")
        store = SessionStore(persistent_path=str(path))
        assert store.get_meta("weird")["kind"] == "chat"


def test_valid_kinds_constant() -> None:
    """If we ever add a third kind (e.g. ``"channel"``) the test that
    breaks first should be this one — forces us to think about
    backwards-compat for the file format."""
    assert VALID_SESSION_KINDS == ("chat", "rp")


# ─── Gateway-level tests ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def _register(client: TestClient, session_id: str) -> None:
    resp = client.post(
        "/api/v1/sessions/register",
        json={"session_id": session_id, "project_id": "default"},
    )
    assert resp.status_code == 200, resp.text


def _patch(client: TestClient, session_id: str, **fields) -> dict:
    resp = client.patch(
        f"/api/v1/sessions/{session_id}", json=fields,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_gateway_lists_sessions_with_kind(lexy_client: TestClient) -> None:
    sid = "test-9.12-list-kind"
    try:
        _register(lexy_client, sid)
        resp = lexy_client.get("/api/v1/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        ours = [s for s in sessions if s["id"] == sid]
        assert ours, "newly registered session not in list"
        assert ours[0]["kind"] == "chat"
    finally:
        lexy_client.delete(f"/api/v1/sessions/{sid}")


def test_gateway_kind_filter_excludes_other_kind(
    lexy_client: TestClient,
) -> None:
    chat_sid = "test-9.12-filter-chat"
    rp_sid = "test-9.12-filter-rp"
    try:
        _register(lexy_client, chat_sid)
        _register(lexy_client, rp_sid)
        _patch(lexy_client, rp_sid, kind="rp")

        # ?kind=rp must return rp_sid but NOT chat_sid
        resp_rp = lexy_client.get("/api/v1/sessions?kind=rp")
        rp_ids = {s["id"] for s in resp_rp.json()["sessions"]}
        assert rp_sid in rp_ids
        assert chat_sid not in rp_ids

        # ?kind=chat must return chat_sid but NOT rp_sid
        resp_chat = lexy_client.get("/api/v1/sessions?kind=chat")
        chat_ids = {s["id"] for s in resp_chat.json()["sessions"]}
        assert chat_sid in chat_ids
        assert rp_sid not in chat_ids

        # No filter → both visible
        resp_all = lexy_client.get("/api/v1/sessions")
        all_ids = {s["id"] for s in resp_all.json()["sessions"]}
        assert {chat_sid, rp_sid} <= all_ids
    finally:
        lexy_client.delete(f"/api/v1/sessions/{chat_sid}")
        lexy_client.delete(f"/api/v1/sessions/{rp_sid}")


def test_gateway_kind_filter_invalid_returns_400(
    lexy_client: TestClient,
) -> None:
    resp = lexy_client.get("/api/v1/sessions?kind=galactic")
    assert resp.status_code == 400


def test_gateway_history_returns_kind(lexy_client: TestClient) -> None:
    sid = "test-9.12-history-kind"
    try:
        _register(lexy_client, sid)
        _patch(lexy_client, sid, kind="rp")
        resp = lexy_client.get(f"/api/v1/sessions/{sid}/history")
        assert resp.status_code == 200
        assert resp.json()["kind"] == "rp"
    finally:
        lexy_client.delete(f"/api/v1/sessions/{sid}")


def test_gateway_patch_kind_invalid_returns_400(
    lexy_client: TestClient,
) -> None:
    sid = "test-9.12-bad-kind"
    try:
        _register(lexy_client, sid)
        resp = lexy_client.patch(
            f"/api/v1/sessions/{sid}", json={"kind": "galactic"}
        )
        assert resp.status_code == 400
    finally:
        lexy_client.delete(f"/api/v1/sessions/{sid}")


def test_gateway_patch_kind_idempotent(lexy_client: TestClient) -> None:
    """Patching with the *same* kind shouldn't error out — the route
    just reports an empty ``changes`` map. Important because the
    frontend may PATCH redundantly when it's not sure of state."""
    sid = "test-9.12-idempotent"
    try:
        _register(lexy_client, sid)
        body = _patch(lexy_client, sid, kind="rp")
        assert body["changes"].get("kind") == "rp"
        # Same call again — no-op.
        body2 = _patch(lexy_client, sid, kind="rp")
        assert "kind" not in body2["changes"]
    finally:
        lexy_client.delete(f"/api/v1/sessions/{sid}")
