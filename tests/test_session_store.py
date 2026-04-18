"""Tests for the in-memory SessionStore."""

from __future__ import annotations

from lexy_core.agent import SessionStore


def test_append_and_get() -> None:
    store = SessionStore(max_messages=20)
    store.append("s1", "user", "hi")
    store.append("s1", "assistant", "hello")
    history = store.get("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hi"}
    assert history[1] == {"role": "assistant", "content": "hello"}


def test_append_pair() -> None:
    store = SessionStore()
    store.append_pair("s1", "wetter in hechthausen?", "9.1°C, klar")
    assert store.length("s1") == 2
    history = store.get("s1")
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_bounded_window_drops_oldest() -> None:
    store = SessionStore(max_messages=4)
    store.append_pair("s1", "q1", "a1")
    store.append_pair("s1", "q2", "a2")
    store.append_pair("s1", "q3", "a3")  # pushes q1/a1 out

    history = store.get("s1")
    assert len(history) == 4
    assert history[0]["content"] == "q2"
    assert history[-1]["content"] == "a3"
    # Window always starts on a user turn after trimming
    assert history[0]["role"] == "user"


def test_multiple_sessions_isolated() -> None:
    store = SessionStore()
    store.append_pair("alice", "hi", "hey")
    store.append_pair("bob", "hello", "moin")
    assert store.length("alice") == 2
    assert store.length("bob") == 2
    assert store.get("alice")[0]["content"] == "hi"
    assert store.get("bob")[0]["content"] == "hello"


def test_clear_session() -> None:
    store = SessionStore()
    store.append_pair("s1", "q", "a")
    store.append_pair("s1", "q2", "a2")
    dropped = store.clear("s1")
    assert dropped == 4
    assert store.length("s1") == 0
    # Clearing an unknown session is a no-op
    assert store.clear("nope") == 0


def test_get_returns_copy() -> None:
    store = SessionStore()
    store.append_pair("s1", "q", "a")
    h1 = store.get("s1")
    h1[0]["content"] = "MUTATED"
    h2 = store.get("s1")
    assert h2[0]["content"] == "q"


def test_empty_session_id_ignored() -> None:
    store = SessionStore()
    store.append("", "user", "hi")
    store.append_pair("", "hi", "hello")
    assert store.sessions() == []
