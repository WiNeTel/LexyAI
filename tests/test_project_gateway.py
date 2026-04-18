"""
HTTP gateway tests for the project + session-move endpoints.

We boot a real LexyApp (same pattern as test_gateway.py) and run the
endpoints through ``fastapi.testclient.TestClient``. The module-level
fixture cleans up any projects we created so we don't pollute the shared
``data/projects.json``.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


_TEST_PREFIX = "__pytest_project_gateway__"


@pytest.fixture(scope="module")
def lexy_client() -> Iterator[TestClient]:
    """Boot a LexyApp once for all project-gateway tests."""
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    client._lexy = app  # type: ignore[attr-defined]
    yield client
    # Purge any projects we created in this module
    for project in app.project_store.list(include_archived=True):
        if project.name.startswith(_TEST_PREFIX):
            app.project_store.delete(project.id)
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def _unique_name(label: str) -> str:
    return f"{_TEST_PREFIX}{label}-{uuid.uuid4().hex[:6]}"


# ─── List / get default ──────────────────────────────────────────────────────


def test_list_projects_contains_default(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert "projects" in data
    ids = {p["id"] for p in data["projects"]}
    assert "default" in ids


def test_get_default_project(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/projects/default")
    assert resp.status_code == 200
    project = resp.json()["project"]
    assert project["is_default"] is True
    assert project["name"]


def test_get_unknown_project_returns_404(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/projects/does-not-exist")
    assert resp.status_code == 404


# ─── Create / update / delete ────────────────────────────────────────────────


def test_create_project(lexy_client: TestClient) -> None:
    name = _unique_name("create")
    resp = lexy_client.post(
        "/api/v1/projects",
        json={
            "name": name,
            "description": "Test project",
            "color": "#112233",
            "icon": "🎮",
            "memory_scoped": True,
        },
    )
    assert resp.status_code == 200
    project = resp.json()["project"]
    assert project["name"] == name
    assert project["color"] == "#112233"
    assert project["icon"] == "🎮"
    assert project["memory_scoped"] is True
    assert project["is_default"] is False


def test_update_project(lexy_client: TestClient) -> None:
    created = lexy_client.post(
        "/api/v1/projects",
        json={"name": _unique_name("upd")},
    ).json()["project"]

    resp = lexy_client.patch(
        f"/api/v1/projects/{created['id']}",
        json={"color": "#abcdef", "icon": "📚"},
    )
    assert resp.status_code == 200
    updated = resp.json()["project"]
    assert updated["color"] == "#abcdef"
    assert updated["icon"] == "📚"
    # Name was NOT in the patch so it must be untouched
    assert updated["name"] == created["name"]


def test_update_unknown_project_returns_404(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/projects/ghost", json={"name": "x"}
    )
    assert resp.status_code == 404


def test_delete_project_returns_migrated_count(lexy_client: TestClient) -> None:
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    created = lexy_client.post(
        "/api/v1/projects", json={"name": _unique_name("del")}
    ).json()["project"]

    # Attach a couple of sessions to it via SessionStore directly
    app.session_store.register_empty("s_del_1", project_id=created["id"])
    app.session_store.register_empty("s_del_2", project_id=created["id"])

    resp = lexy_client.delete(f"/api/v1/projects/{created['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert data["migrated_sessions"] == 2

    # Sessions should now point at the default project
    for sid in ("s_del_1", "s_del_2"):
        meta = app.session_store.get_meta(sid)
        assert meta.get("project_id") == "default"
        app.session_store.clear(sid)


def test_delete_default_project_is_blocked(lexy_client: TestClient) -> None:
    resp = lexy_client.delete("/api/v1/projects/default")
    assert resp.status_code == 400


# ─── Archive / unarchive ─────────────────────────────────────────────────────


def test_archive_hides_from_default_listing(lexy_client: TestClient) -> None:
    created = lexy_client.post(
        "/api/v1/projects", json={"name": _unique_name("arch")}
    ).json()["project"]

    resp = lexy_client.post(
        f"/api/v1/projects/{created['id']}/archive"
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "archived"

    listed = lexy_client.get("/api/v1/projects").json()["projects"]
    assert all(p["id"] != created["id"] for p in listed)

    full = lexy_client.get(
        "/api/v1/projects?include_archived=true"
    ).json()["projects"]
    assert any(p["id"] == created["id"] for p in full)

    unarchive = lexy_client.post(
        f"/api/v1/projects/{created['id']}/unarchive"
    )
    assert unarchive.status_code == 200
    assert unarchive.json()["status"] == "unarchived"


# ─── Session → Project move + filter ─────────────────────────────────────────


def test_session_patch_moves_between_projects(lexy_client: TestClient) -> None:
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    created = lexy_client.post(
        "/api/v1/projects", json={"name": _unique_name("move")}
    ).json()["project"]

    sid = f"s_move_{uuid.uuid4().hex[:6]}"
    app.session_store.register_empty(sid, project_id="default")

    resp = lexy_client.patch(
        f"/api/v1/sessions/{sid}",
        json={"project_id": created["id"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["changes"]["project_id"] == created["id"]

    meta = app.session_store.get_meta(sid)
    assert meta.get("project_id") == created["id"]
    app.session_store.clear(sid)


def test_session_patch_rejects_unknown_project(lexy_client: TestClient) -> None:
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    sid = f"s_bad_{uuid.uuid4().hex[:6]}"
    app.session_store.register_empty(sid, project_id="default")

    resp = lexy_client.patch(
        f"/api/v1/sessions/{sid}",
        json={"project_id": "nonexistent-xyz"},
    )
    assert resp.status_code == 404
    app.session_store.clear(sid)


def test_session_patch_title_only(lexy_client: TestClient) -> None:
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]
    sid = f"s_t_{uuid.uuid4().hex[:6]}"
    app.session_store.register_empty(sid)

    resp = lexy_client.patch(
        f"/api/v1/sessions/{sid}",
        json={"title": "Manual title"},
    )
    assert resp.status_code == 200
    assert resp.json()["changes"]["title"] == "Manual title"
    assert app.session_store.get_meta(sid).get("title") == "Manual title"
    app.session_store.clear(sid)


def test_session_patch_unknown_session_404(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/sessions/ghost-session", json={"title": "x"}
    )
    assert resp.status_code == 404


def test_sessions_list_filters_by_project(lexy_client: TestClient) -> None:
    app: LexyApp = lexy_client._lexy  # type: ignore[attr-defined]

    created = lexy_client.post(
        "/api/v1/projects", json={"name": _unique_name("filter")}
    ).json()["project"]

    sid_in = f"s_in_{uuid.uuid4().hex[:6]}"
    sid_out = f"s_out_{uuid.uuid4().hex[:6]}"
    app.session_store.register_empty(sid_in, project_id=created["id"])
    app.session_store.register_empty(sid_out, project_id="default")

    resp = lexy_client.get(
        f"/api/v1/sessions?project_id={created['id']}"
    )
    assert resp.status_code == 200
    ids = {s["id"] for s in resp.json()["sessions"]}
    assert sid_in in ids
    assert sid_out not in ids

    # Default project also surfaces sessions without explicit project_id
    resp_default = lexy_client.get("/api/v1/sessions?project_id=default")
    default_ids = {s["id"] for s in resp_default.json()["sessions"]}
    assert sid_out in default_ids

    app.session_store.clear(sid_in)
    app.session_store.clear(sid_out)


# ─── Validation & edge cases ─────────────────────────────────────────────────


def test_create_rejects_empty_name(lexy_client: TestClient) -> None:
    resp = lexy_client.post("/api/v1/projects", json={"name": "   "})
    assert resp.status_code in (400, 422)


def test_create_normalises_bad_color(lexy_client: TestClient) -> None:
    resp = lexy_client.post(
        "/api/v1/projects",
        json={"name": _unique_name("badcolor"), "color": "not-a-color"},
    )
    assert resp.status_code == 200
    project = resp.json()["project"]
    # Validator silently falls back to the default palette color
    assert project["color"] == "#7aa2f7"
