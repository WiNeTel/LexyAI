"""
Phase 11 — REST-layer tests for the skill_writer import/export routes.

Boots a real :class:`LexyApp` and drives the new REST endpoints
through the FastAPI :class:`TestClient`. Covers:

* import-zip happy path (REST → registry visible)
* invalid spec → 400 with structured detail
* conflict (skill exists + overwrite=False) → 409
* overwrite=True → 200, registry updated
* export-zip → returns valid ZIP bytes a fresh import accepts
* round-trip via REST: export → import (with overwrite) preserves card
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def _build_minimal_zip(name: str = "rest-test-skill") -> bytes:
    """Build a minimal valid skill ZIP entirely in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            f"{name}/SKILL.md",
            f"---\nname: {name}\ndescription: REST-test skill.\n---\n",
        )
        zf.writestr(
            f"{name}/scripts/skill.py",
            (
                "from typing import Any\n\n"
                "async def execute(api: Any, **kwargs: Any) -> dict[str, Any]:\n"
                '    return {"ok": True}\n'
            ),
        )
    return buf.getvalue()


def _cleanup(client: TestClient, name: str) -> None:
    """Best-effort cleanup so cross-test state doesn't bleed.

    The skill_writer plugin doesn't expose a REST DELETE endpoint, so
    we live with the leftovers under ``data/skills/`` for now —
    subsequent tests use unique names to avoid collisions.
    """
    pass  # placeholder, names below are unique per test


# ─── Import path ─────────────────────────────────────────────────────


def test_import_skill_happy_path(lexy_client: TestClient) -> None:
    name = "rest-import-happy"
    payload = _build_minimal_zip(name=name)

    resp = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["overwrote_existing"] is False
    assert body["skill"]["name"] == name


def test_import_invalid_frontmatter_returns_400(
    lexy_client: TestClient,
) -> None:
    """SKILL.md with an uppercase name violates the spec."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "BadName/SKILL.md",
            "---\nname: BadName\ndescription: D.\n---\n",
        )
    resp = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": ("bad.zip", buf.getvalue(), "application/zip")},
    )
    assert resp.status_code == 400
    assert "validation" in resp.json()["detail"].lower()


def test_import_corrupted_zip_returns_400(lexy_client: TestClient) -> None:
    resp = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": ("garbage.zip", b"not a zip", "application/zip")},
    )
    assert resp.status_code == 400


def test_import_empty_upload_returns_400(lexy_client: TestClient) -> None:
    resp = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": ("empty.zip", b"", "application/zip")},
    )
    assert resp.status_code == 400


def test_import_conflict_returns_409(lexy_client: TestClient) -> None:
    name = "rest-import-conflict"
    payload = _build_minimal_zip(name=name)
    # First import succeeds.
    resp1 = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
    )
    assert resp1.status_code == 200, resp1.text
    # Second import without overwrite=true → 409.
    resp2 = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
    )
    assert resp2.status_code == 409


def test_import_overwrite_flag(lexy_client: TestClient) -> None:
    name = "rest-import-overwrite"
    payload = _build_minimal_zip(name=name)
    lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
    )
    # With overwrite=true the second import succeeds.
    resp = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
        data={"overwrite": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["overwrote_existing"] is True


# ─── Export path ─────────────────────────────────────────────────────


def test_export_unknown_returns_404(lexy_client: TestClient) -> None:
    resp = lexy_client.get(
        "/api/v1/plugins/skill_writer/skills/__no-such-skill__/export"
    )
    assert resp.status_code == 404


def test_export_returns_zip_bytes(lexy_client: TestClient) -> None:
    name = "rest-export-test"
    # Ensure something to export.
    payload = _build_minimal_zip(name=name)
    lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
    )
    resp = lexy_client.get(
        f"/api/v1/plugins/skill_writer/skills/{name}/export"
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    # Content is a valid ZIP with the expected layout.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
    assert f"{name}/SKILL.md" in names
    assert f"{name}/scripts/skill.py" in names


# ─── Round-trip ─────────────────────────────────────────────────────


def test_export_then_import_roundtrip(lexy_client: TestClient) -> None:
    name = "rest-roundtrip"
    payload = _build_minimal_zip(name=name)
    # Import original.
    lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", payload, "application/zip")},
    )
    # Export back.
    exported = lexy_client.get(
        f"/api/v1/plugins/skill_writer/skills/{name}/export"
    )
    assert exported.status_code == 200
    # Re-import with overwrite — must succeed.
    resp = lexy_client.post(
        "/api/v1/plugins/skill_writer/skills/import",
        files={"file": (f"{name}.zip", exported.content, "application/zip")},
        data={"overwrite": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["skill"]["name"] == name
