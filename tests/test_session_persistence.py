"""Tests for SessionStore JSON-on-disk persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexy_core.agent import SessionStore


def test_save_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = SessionStore(max_messages=10, persistent_path=path)
    store.append_pair("s1", "hi", "hello")

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    assert "s1" in data["sessions"]
    assert data["sessions"]["s1"]["messages"][0]["content"] == "hi"
    assert data["sessions"]["s1"]["messages"][1]["content"] == "hello"


def test_load_restores_sessions(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    first = SessionStore(max_messages=10, persistent_path=path)
    first.append_pair("alice", "q1", "a1")
    first.append_pair("bob", "q2", "a2")

    # New instance reads the same file
    second = SessionStore(max_messages=10, persistent_path=path)
    assert set(second.sessions()) == {"alice", "bob"}
    assert second.get("alice")[0]["content"] == "q1"
    assert second.get("bob")[1]["content"] == "a2"


def test_missing_file_is_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    store = SessionStore(persistent_path=path)
    assert store.sessions() == []
    # save() then creates the file
    store.append("s1", "user", "hi")
    assert path.exists()


def test_corrupt_file_is_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text("this is {{ not json", encoding="utf-8")
    store = SessionStore(persistent_path=path)
    # Should not raise, just start empty
    assert store.sessions() == []


def test_invalid_shape_is_empty_store(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"sessions": "not a dict"}), encoding="utf-8")
    store = SessionStore(persistent_path=path)
    assert store.sessions() == []


def test_regenerate_pop_persists(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = SessionStore(max_messages=10, persistent_path=path)
    store.append_pair("s1", "q1", "a1")
    store.append_pair("s1", "q2", "a2")
    user_msg, asst_msg = store.pop_last_pair("s1")
    assert user_msg is not None
    assert asst_msg is not None

    # Fresh instance should only see the first pair
    reloaded = SessionStore(max_messages=10, persistent_path=path)
    history = reloaded.get("s1")
    assert len(history) == 2
    assert history[-1]["content"] == "a1"


def test_edit_delete_persist(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = SessionStore(max_messages=10, persistent_path=path)
    store.append_pair("s1", "original", "answer")
    store.replace_at("s1", 0, "edited")
    store.delete_at("s1", 1)

    reloaded = SessionStore(max_messages=10, persistent_path=path)
    history = reloaded.get("s1")
    assert len(history) == 1
    assert history[0]["content"] == "edited"


def test_clear_persists(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = SessionStore(max_messages=10, persistent_path=path)
    store.append_pair("keep", "k", "k")
    store.append_pair("drop", "d", "d")
    store.clear("drop")

    reloaded = SessionStore(max_messages=10, persistent_path=path)
    assert reloaded.sessions() == ["keep"]


def test_reset_all_persists(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = SessionStore(max_messages=10, persistent_path=path)
    store.append_pair("s1", "q", "a")
    store.append_pair("s2", "q", "a")
    store.reset_all()

    reloaded = SessionStore(max_messages=10, persistent_path=path)
    assert reloaded.sessions() == []


def test_no_path_means_no_file(tmp_path: Path) -> None:
    store = SessionStore(max_messages=10)  # no path
    store.append_pair("s1", "q", "a")
    # No file gets written anywhere
    assert store.save() is False
    # Directory still empty
    assert list(tmp_path.iterdir()) == []


def test_trim_on_load(tmp_path: Path) -> None:
    """A file containing more messages than max_messages is trimmed on load."""
    path = tmp_path / "sessions.json"
    payload = {
        "version": 1,
        "sessions": {
            "s1": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
                {"role": "assistant", "content": "a3"},
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = SessionStore(max_messages=4, persistent_path=path)
    history = store.get("s1")
    assert len(history) == 4
    assert history[0]["content"] == "q2"
    assert history[-1]["content"] == "a3"


def test_invalid_entries_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    payload = {
        "sessions": {
            "s1": [
                {"role": "user", "content": "keep me"},
                "not a dict",
                {"role": 123, "content": "bad role"},
                {"role": "assistant", "content": None},
                {"role": "assistant", "content": "also kept"},
            ],
            "empty_session": [],
            42: [{"role": "user", "content": "bad key"}],  # non-str key
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = SessionStore(max_messages=20, persistent_path=path)
    history = store.get("s1")
    assert [m["content"] for m in history] == ["keep me", "also kept"]
    # Empty session was dropped entirely
    assert "empty_session" not in store.sessions()


def test_atomic_save_no_leftover_tmp(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = SessionStore(persistent_path=path)
    store.append_pair("s1", "q", "a")
    store.append_pair("s1", "q2", "a2")
    # .tmp file should have been replaced, not left behind
    tmp_files = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tmp_files == []
    assert path.exists()


@pytest.mark.asyncio
async def test_lexy_app_restart_preserves_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Booting LexyApp twice against the same sessions_path keeps history."""
    from lexy_core.config import LexyConfig, SystemConfig

    sessions_path = tmp_path / "sessions.json"

    # Seed a sessions.json via a plain SessionStore (simulates a previous run)
    seed = SessionStore(max_messages=20, persistent_path=sessions_path)
    seed.append_pair("persisted", "remember me?", "of course")
    assert sessions_path.exists()

    # Build a minimal config and feed it to a fresh SessionStore (what
    # LexyApp.startup() does — we don't boot the full app here because
    # that would pull in ChromaDB, embeddings, and uvicorn.)
    cfg = LexyConfig(
        system=SystemConfig(
            sessions_path=str(sessions_path),
            sessions_max_messages=20,
        )
    )
    restored = SessionStore(
        max_messages=cfg.system.sessions_max_messages,
        persistent_path=cfg.system.sessions_path,
    )
    assert restored.sessions() == ["persisted"]
    history = restored.get("persisted")
    assert history[0]["content"] == "remember me?"
    assert history[1]["content"] == "of course"
