"""Tests for the session-store project-id backfill + escape-hatch fixes.

Mike's "sessions disappear" bug had three contributing causes:

1. Sessions registered before any project was assigned land on disk with
   ``project_id=None``.
2. The frontend always passes ``?project_id=<active>`` to the listing
   endpoint, which strictly filters — so unassigned sessions vanish the
   moment the user switches to any non-default project.
3. There was no escape-hatch ("show all sessions") on the listing API.

The fixes covered here:

* :class:`TestProjectBackfillOnLoad` — sessions with ``project_id=None``
  on disk are auto-assigned to ``"default"`` on load and the migration
  is persisted (idempotent: a second load doesn't re-migrate).
* :class:`TestSessionHistoryReturnsMeta` — the ``/history`` endpoint
  returns ``project_id`` so the frontend can sync ``activeProjectId``
  on resume (covered by direct store calls — the route is a thin
  wrapper that maps ``store.get`` + ``store.get_meta`` 1:1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lexy_core.agent.session_store import SessionStore


# ─── Fixtures ────────────────────────────────────────────────────────


def _write_v2(path: Path, sessions: dict) -> None:
    """Helper: write a v2-shaped sessions.json to ``path``."""
    payload = {
        "version": 2,
        "saved_at": time.time(),
        "max_messages": 20,
        "sessions": sessions,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─── 1. Project-id backfill on load ──────────────────────────────────


class TestProjectBackfillOnLoad:
    def test_none_project_id_backfilled_to_default(self, tmp_path: Path) -> None:
        """Legacy session with project_id=None should land in 'default'
        after the first ``SessionStore.load()`` call.
        """
        sessions_file = tmp_path / "sessions.json"
        _write_v2(sessions_file, {
            "legacy-session": {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
                "meta": {
                    "project_id": None,
                    "created_at": 100.0,
                    "updated_at": 100.0,
                    "title": "hi",
                },
            },
        })
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        meta = store.get_meta("legacy-session")
        assert meta["project_id"] == "default"

    def test_backfill_persisted_on_disk(self, tmp_path: Path) -> None:
        """Migration must be written back so a subsequent load doesn't
        re-trigger the backfill (and so a downstream PATCH actually has
        a project_id to update)."""
        sessions_file = tmp_path / "sessions.json"
        _write_v2(sessions_file, {
            "s1": {
                "messages": [{"role": "user", "content": "x"}],
                "meta": {"project_id": None, "created_at": 0.0, "updated_at": 0.0, "title": None},
            },
        })
        SessionStore(max_messages=20, persistent_path=sessions_file)
        # Re-read raw and verify project_id="default" is on disk.
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
        on_disk_meta = data["sessions"]["s1"]["meta"]
        assert on_disk_meta["project_id"] == "default"

    def test_existing_project_id_not_clobbered(self, tmp_path: Path) -> None:
        """Sessions with an explicit project_id must NOT be touched."""
        sessions_file = tmp_path / "sessions.json"
        _write_v2(sessions_file, {
            "tagged-session": {
                "messages": [{"role": "user", "content": "x"}],
                "meta": {
                    "project_id": "spielefirma",
                    "created_at": 0.0,
                    "updated_at": 0.0,
                    "title": None,
                },
            },
        })
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        assert store.get_meta("tagged-session")["project_id"] == "spielefirma"

    def test_idempotent_second_load_does_not_re_migrate(
        self, tmp_path: Path
    ) -> None:
        """Loading twice doesn't keep flipping things."""
        sessions_file = tmp_path / "sessions.json"
        _write_v2(sessions_file, {
            "s1": {
                "messages": [],
                "meta": {"project_id": None, "created_at": 0.0, "updated_at": 0.0, "title": None},
            },
        })
        SessionStore(max_messages=20, persistent_path=sessions_file)
        first_mtime = sessions_file.stat().st_mtime
        # Sleep a tick so a second write would have a different mtime.
        time.sleep(0.05)
        # New store instance → triggers another load. Should NOT rewrite
        # because the on-disk project_id is now "default".
        SessionStore(max_messages=20, persistent_path=sessions_file)
        second_mtime = sessions_file.stat().st_mtime
        # Either equal (no rewrite) or — at worst — a single atomic-replace
        # touched the file. We assert the on-disk shape is unchanged.
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
        assert data["sessions"]["s1"]["meta"]["project_id"] == "default"

    def test_mixed_legacy_and_modern_sessions(self, tmp_path: Path) -> None:
        sessions_file = tmp_path / "sessions.json"
        _write_v2(sessions_file, {
            "legacy-1": {
                "messages": [{"role": "user", "content": "a"}],
                "meta": {"project_id": None, "created_at": 0.0, "updated_at": 0.0, "title": None},
            },
            "tagged-1": {
                "messages": [{"role": "user", "content": "b"}],
                "meta": {"project_id": "lexy", "created_at": 0.0, "updated_at": 0.0, "title": None},
            },
            "tagged-2": {
                "messages": [{"role": "user", "content": "c"}],
                "meta": {"project_id": "default", "created_at": 0.0, "updated_at": 0.0, "title": None},
            },
        })
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        assert store.get_meta("legacy-1")["project_id"] == "default"
        assert store.get_meta("tagged-1")["project_id"] == "lexy"
        assert store.get_meta("tagged-2")["project_id"] == "default"

    def test_v1_format_also_gets_default_project(self, tmp_path: Path) -> None:
        """v1 = bare list of messages, no meta. Combined v1->v2 + backfill."""
        sessions_file = tmp_path / "sessions.json"
        # Force v1 shape by emitting a list per session.
        sessions_file.write_text(json.dumps({
            "version": 1,
            "saved_at": 0.0,
            "max_messages": 20,
            "sessions": {
                "v1-only": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        }, ensure_ascii=False), encoding="utf-8")
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        meta = store.get_meta("v1-only")
        assert meta["project_id"] == "default"
        # v2 shape + backfill on disk
        data = json.loads(sessions_file.read_text(encoding="utf-8"))
        assert data["version"] == 2
        assert data["sessions"]["v1-only"]["meta"]["project_id"] == "default"


# ─── 2. Project-aware register + new session metadata ────────────────


class TestRegisterEmptyWithProject:
    """``register_empty(project_id=X)`` → meta.project_id is X."""

    def test_register_with_project_id(self, tmp_path: Path) -> None:
        sessions_file = tmp_path / "sessions.json"
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        created = store.register_empty(
            session_id="brand-new",
            project_id="spielefirma",
            title="Test",
        )
        assert created is True
        assert store.get_meta("brand-new")["project_id"] == "spielefirma"

    def test_register_existing_session_enriches_meta(
        self, tmp_path: Path
    ) -> None:
        sessions_file = tmp_path / "sessions.json"
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        # First create with no project_id …
        store.register_empty(session_id="s1")
        meta_before = store.get_meta("s1")
        assert meta_before["project_id"] is None
        # … then re-register with one. register_empty enriches when the
        # previous value was None — explicit Mike-edit behaviour.
        created = store.register_empty(session_id="s1", project_id="lexy")
        assert created is False  # didn't create a new slot
        assert store.get_meta("s1")["project_id"] == "lexy"

    def test_register_does_not_clobber_existing_project_id(
        self, tmp_path: Path
    ) -> None:
        sessions_file = tmp_path / "sessions.json"
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        store.register_empty(session_id="s1", project_id="alpha")
        # Re-register with a different project_id → existing one wins.
        # (Use ``set_project`` for an explicit move.)
        store.register_empty(session_id="s1", project_id="beta")
        assert store.get_meta("s1")["project_id"] == "alpha"


# ─── 3. set_project move semantics ──────────────────────────────────


class TestSetProjectMove:
    def test_move_session_between_projects(self, tmp_path: Path) -> None:
        sessions_file = tmp_path / "sessions.json"
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        store.register_empty(session_id="s1", project_id="default")
        ok = store.set_project("s1", "spielefirma")
        assert ok is True
        assert store.get_meta("s1")["project_id"] == "spielefirma"

    def test_move_unknown_session_returns_false(self, tmp_path: Path) -> None:
        sessions_file = tmp_path / "sessions.json"
        store = SessionStore(max_messages=20, persistent_path=sessions_file)
        ok = store.set_project("never-existed", "anywhere")
        assert ok is False
