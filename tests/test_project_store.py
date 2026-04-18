"""Tests for the ProjectStore + Project model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lexy_core.project import (
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    Project,
    ProjectStore,
)


# ─── Project model ──────────────────────────────────────────────────────────


def test_project_default_color_when_invalid() -> None:
    project = Project(id="x", name="Test", color="not-a-color")
    assert project.color == "#7aa2f7"


def test_project_lowercases_hex_color() -> None:
    project = Project(id="x", name="Test", color="#AABBCC")
    assert project.color == "#aabbcc"


def test_project_strips_whitespace_in_name() -> None:
    project = Project(id="x", name="   Hello    World  ")
    assert project.name == "Hello World"


def test_project_rejects_blank_name() -> None:
    with pytest.raises(Exception):
        Project(id="x", name="     ")


def test_project_to_dict_round_trips() -> None:
    project = Project(id="abc", name="Test", color="#ff0000", icon="🎮")
    blob = project.to_dict()
    rebuilt = Project(**blob)
    assert rebuilt == project


# ─── Bootstrap + persistence ─────────────────────────────────────────────────


def test_fresh_store_creates_default(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    store = ProjectStore(persistent_path=path)
    default = store.get_default()
    assert default.id == DEFAULT_PROJECT_ID
    assert default.name == DEFAULT_PROJECT_NAME
    assert default.is_default is True
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert DEFAULT_PROJECT_ID in data["projects"]


def test_load_restores_projects(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    first = ProjectStore(persistent_path=path)
    created = first.create(name="Spielefirma", color="#ff00aa", icon="🎮")
    assert created.id != DEFAULT_PROJECT_ID

    second = ProjectStore(persistent_path=path)
    loaded = second.get(created.id)
    assert loaded is not None
    assert loaded.name == "Spielefirma"
    assert loaded.color == "#ff00aa"
    assert loaded.icon == "🎮"


def test_load_with_missing_file_creates_default(tmp_path: Path) -> None:
    path = tmp_path / "absent.json"
    store = ProjectStore(persistent_path=path)
    assert store.exists(DEFAULT_PROJECT_ID)


def test_load_corrupt_file_recovers(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = ProjectStore(persistent_path=path)
    # Should still bootstrap default
    assert store.exists(DEFAULT_PROJECT_ID)


def test_load_invalid_shape_recovers(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    path.write_text(json.dumps({"version": 1, "projects": "garbage"}), encoding="utf-8")
    store = ProjectStore(persistent_path=path)
    assert store.exists(DEFAULT_PROJECT_ID)


def test_load_skips_invalid_entries(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    payload = {
        "version": 1,
        "projects": {
            "ok": {
                "id": "ok",
                "name": "Good",
                "color": "#abcdef",
                "icon": "",
                "persona_override": "",
                "memory_scoped": True,
                "is_default": False,
                "archived": False,
                "created_at": 1.0,
                "updated_at": 1.0,
            },
            "broken": {"missing": "fields"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = ProjectStore(persistent_path=path)
    assert store.exists("ok")
    assert not store.exists("broken")
    # Default still gets bootstrapped
    assert store.exists(DEFAULT_PROJECT_ID)


def test_id_in_payload_overridden_by_dict_key(tmp_path: Path) -> None:
    path = tmp_path / "projects.json"
    payload = {
        "version": 1,
        "projects": {
            "key-id": {
                "id": "wrong-id",
                "name": "Misaligned",
                "color": "#7aa2f7",
                "icon": "",
                "persona_override": "",
                "memory_scoped": True,
                "is_default": False,
                "archived": False,
                "created_at": 1.0,
                "updated_at": 1.0,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = ProjectStore(persistent_path=path)
    project = store.get("key-id")
    assert project is not None
    assert project.id == "key-id"


# ─── CRUD ────────────────────────────────────────────────────────────────────


def test_create_returns_unique_id(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    a = store.create(name="A")
    b = store.create(name="B")
    assert a.id != b.id
    assert a.id != DEFAULT_PROJECT_ID


def test_update_changes_fields(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    project = store.create(name="Old")
    updated = store.update(project.id, name="New", color="#112233", icon="✨")
    assert updated is not None
    assert updated.name == "New"
    assert updated.color == "#112233"
    assert updated.icon == "✨"
    assert updated.updated_at >= project.created_at


def test_update_ignores_protected_fields(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    project = store.create(name="X")
    updated = store.update(
        project.id, id="hijacked", is_default=True, created_at=0.0
    )
    assert updated is not None
    assert updated.id == project.id
    assert updated.is_default is False
    assert updated.created_at == project.created_at


def test_update_invalid_color_falls_back(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    project = store.create(name="X", color="#000000")
    updated = store.update(project.id, color="not-hex")
    assert updated is not None
    assert updated.color == "#7aa2f7"


def test_update_unknown_id(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    assert store.update("ghost", name="Boo") is None


def test_default_project_protected_from_delete(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    deleted, _ = store.delete(DEFAULT_PROJECT_ID)
    assert deleted is False
    assert store.exists(DEFAULT_PROJECT_ID)


def test_delete_returns_snapshot(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    project = store.create(name="Doomed")
    deleted, snapshot = store.delete(project.id)
    assert deleted is True
    assert snapshot is not None
    assert snapshot.name == "Doomed"
    assert not store.exists(project.id)


def test_archive_and_unarchive(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    project = store.create(name="Old Stuff")
    assert store.archive(project.id) is True
    visible = store.list(include_archived=False)
    assert all(p.id != project.id for p in visible)
    full = store.list(include_archived=True)
    assert any(p.id == project.id and p.archived for p in full)
    assert store.unarchive(project.id) is True
    assert any(p.id == project.id for p in store.list())


def test_archive_default_blocked(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    assert store.archive(DEFAULT_PROJECT_ID) is False


def test_archive_idempotent(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    project = store.create(name="X")
    assert store.archive(project.id) is True
    assert store.archive(project.id) is False  # already archived


# ─── Listing ─────────────────────────────────────────────────────────────────


def test_list_default_first(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    store.create(name="Other")
    items = store.list()
    assert items[0].id == DEFAULT_PROJECT_ID


def test_list_excludes_archived_by_default(tmp_path: Path) -> None:
    store = ProjectStore(persistent_path=tmp_path / "p.json")
    a = store.create(name="A")
    store.archive(a.id)
    visible = store.list()
    assert all(p.id != a.id for p in visible)
