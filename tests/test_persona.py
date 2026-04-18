"""
Tests for the Persona system:
* Pydantic validation + defaults
* Sectioned prompt assembly
* File round-trip (load/save/reset)
* Migration from monolithic system_prompt
* LexyAgent._plan uses persona.assemble()
* API: GET /api/v1/persona, PATCH, POST /reset

IMPORTANT: API tests redirect persona writes to a temp file so they
never overwrite the user's real ``config/persona.yaml``.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lexy_core.agent import (
    DEFAULT_IDENTITY,
    DEFAULT_RULES,
    DEFAULT_STYLE,
    Persona,
    PersonaSections,
    load_persona,
    save_persona,
    reset_persona,
)
from lexy_core.agent.persona import PROTECTED_CAPABILITIES, PROTECTED_CONTEXT
from lexy_core.app import LexyApp


# ─── Persona model ─────────────────────────────────────────────────────────


def test_persona_defaults() -> None:
    persona = Persona()
    assert persona.name == "Lexy"
    assert persona.user_name == "Mike"
    assert persona.language == "de"
    assert persona.thinking_enabled is True
    assert persona.temperature_override is None
    assert persona.tags == []
    assert persona.sections.identity == DEFAULT_IDENTITY
    assert persona.sections.style == DEFAULT_STYLE
    assert persona.sections.rules == DEFAULT_RULES


def test_assemble_contains_all_sections() -> None:
    """The assembled prompt must include all 5 layers."""
    persona = Persona()
    assembled = persona.assemble()
    # User sections
    assert "## Wer du bist" in assembled
    assert "## Wie du dich ausdrückst" in assembled
    assert "## Was du NICHT machst" in assembled
    # Protected sections
    assert "## Erinnerung & Kontext" in assembled
    assert "## Wie du denkst" in assembled
    assert "## Tools" in assembled
    assert "## Deine erweiterten Fähigkeiten" in assembled


def test_assemble_substitutes_placeholders() -> None:
    persona = Persona(name="Aria", user_name="Mo", language="en")
    assembled = persona.assemble()
    # Protected capabilities section references {user_name}
    assert "Mo" in assembled
    # Should NOT contain raw {user_name}
    assert "{user_name}" not in assembled


def test_system_prompt_property_equals_assemble() -> None:
    persona = Persona()
    assert persona.system_prompt == persona.assemble()


def test_rendered_system_prompt_backward_compat() -> None:
    persona = Persona()
    assert persona.rendered_system_prompt() == persona.assemble()


def test_default_prompt_is_human() -> None:
    """Sanity check that the default personality isn't bot-robotic."""
    assembled = Persona().assemble()
    assert "Lexy" in assembled
    assert "Mike" in assembled
    assert len(assembled) > 400
    assert "natürlich" in assembled.lower() or "natuerlich" in assembled.lower()
    assert "Meinung" in assembled


def test_custom_sections() -> None:
    persona = Persona(
        sections=PersonaSections(
            identity="I am TestBot.",
            style="Be concise.",
            rules="No lying.",
        )
    )
    assembled = persona.assemble()
    assert "I am TestBot." in assembled
    assert "Be concise." in assembled
    assert "No lying." in assembled
    # Protected sections still present
    assert "## Wie du denkst" in assembled


# ─── File round-trip ───────────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "persona.yaml"
    original = Persona(
        name="Nova",
        user_name="Sam",
        language="en",
        thinking_enabled=False,
        sections=PersonaSections(
            identity="You are Nova.",
            style="Friendly.",
            rules="Be kind.",
        ),
        tags=["friendly", "concise"],
    )
    save_persona(original, path)
    assert path.exists()

    loaded = load_persona(path)
    assert loaded.name == "Nova"
    assert loaded.user_name == "Sam"
    assert loaded.language == "en"
    assert loaded.thinking_enabled is False
    assert loaded.sections.identity == "You are Nova."
    assert loaded.sections.style == "Friendly."
    assert loaded.sections.rules == "Be kind."
    assert loaded.tags == ["friendly", "concise"]


def test_load_missing_file_creates_default(tmp_path: Path) -> None:
    path = tmp_path / "not-there.yaml"
    assert not path.exists()
    loaded = load_persona(path)
    assert path.exists()
    assert loaded.name == "Lexy"


def test_load_corrupt_yaml_returns_default(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("not: valid: yaml: here: :", encoding="utf-8")
    loaded = load_persona(path)
    assert loaded.name == "Lexy"


def test_reset_overwrites_file(tmp_path: Path) -> None:
    path = tmp_path / "persona.yaml"
    custom = Persona(
        name="Custom",
        sections=PersonaSections(identity="custom identity"),
    )
    save_persona(custom, path)
    assert load_persona(path).name == "Custom"

    reset = reset_persona(path)
    assert reset.name == "Lexy"
    loaded = load_persona(path)
    assert loaded.sections.identity == DEFAULT_IDENTITY


def test_migration_from_monolithic(tmp_path: Path) -> None:
    """Old-format persona.yaml with system_prompt should auto-migrate."""
    path = tmp_path / "persona.yaml"
    old_content = (
        "name: OldLexy\n"
        "user_name: OldUser\n"
        "language: de\n"
        'system_prompt: "## Wer du bist\\nIch bin alt.\\n\\n'
        "## Wie du dich ausdrückst\\nSei kurz.\\n\\n"
        '## Was du NICHT machst\\nKein Spam."\n'
    )
    path.write_text(old_content, encoding="utf-8")
    loaded = load_persona(path)
    assert loaded.name == "OldLexy"
    # Should have migrated into sections
    assert "alt" in loaded.sections.identity
    assert "kurz" in loaded.sections.style or "Sei kurz" in loaded.sections.style
    assert "Spam" in loaded.sections.rules


# ─── API integration ───────────────────────────────────────────────────────
#
# We redirect PERSONA_PATH to a temporary file so these tests never touch
# the user's real config/persona.yaml. The real file is first copied into
# the temp location so the app loads whatever the user had before.


@pytest.fixture(scope="module")
def _persona_tmp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temp dir with a copy of the real persona.yaml."""
    tmp = tmp_path_factory.mktemp("persona")
    real = Path("config/persona.yaml")
    dest = tmp / "persona.yaml"
    if real.exists():
        shutil.copy2(real, dest)
    return dest


@pytest.fixture(scope="module")
def lexy_client(_persona_tmp_dir: Path) -> TestClient:
    # Patch PERSONA_PATH BEFORE the app loads the persona, so all reads
    # and writes go to the temp file.
    import lexy_core.agent.persona as persona_mod

    original_path = persona_mod.PERSONA_PATH
    persona_mod.PERSONA_PATH = _persona_tmp_dir

    app = LexyApp("config/config.yaml")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    loop.run_until_complete(app.shutdown())

    # Restore
    persona_mod.PERSONA_PATH = original_path


def test_api_get_persona(lexy_client: TestClient) -> None:
    resp = lexy_client.get("/api/v1/persona")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "sections" in data
    assert "system_prompt" in data  # assembled read-only
    assert len(data["system_prompt"]) > 100  # must have actual content
    assert "thinking_enabled" in data


def test_api_patch_persona_sections(lexy_client: TestClient) -> None:
    original = lexy_client.get("/api/v1/persona").json()

    try:
        resp = lexy_client.patch(
            "/api/v1/persona",
            json={
                "name": "TestBot",
                "sections": {"identity": "I am TestBot for unit tests."},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["persona"]["name"] == "TestBot"
        assert data["persona"]["sections"]["identity"] == "I am TestBot for unit tests."
        # Style and rules should be unchanged
        assert data["persona"]["sections"]["style"] == original["sections"]["style"]

        # Verify via re-GET
        verify = lexy_client.get("/api/v1/persona").json()
        assert verify["name"] == "TestBot"
        assert "I am TestBot" in verify["system_prompt"]

        # Verify in-memory persona
        app = lexy_client.app.state.lexy
        assert app.persona.name == "TestBot"
    finally:
        lexy_client.patch(
            "/api/v1/persona",
            json={
                "name": original["name"],
                "sections": original["sections"],
            },
        )


def test_api_patch_thinking_toggle(lexy_client: TestClient) -> None:
    resp = lexy_client.patch(
        "/api/v1/persona",
        json={"thinking_enabled": False},
    )
    assert resp.status_code == 200
    assert resp.json()["persona"]["thinking_enabled"] is False

    # Toggle back
    resp = lexy_client.patch(
        "/api/v1/persona",
        json={"thinking_enabled": True},
    )
    assert resp.status_code == 200
    assert resp.json()["persona"]["thinking_enabled"] is True


def test_api_reset_persona(lexy_client: TestClient) -> None:
    lexy_client.patch(
        "/api/v1/persona",
        json={"name": "Silly", "sections": {"identity": "garbage"}},
    )
    resp = lexy_client.post("/api/v1/persona/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["persona"]["name"] == "Lexy"
    assert data["persona"]["sections"]["identity"] == DEFAULT_IDENTITY


def test_api_patch_noop_returns_no_op(lexy_client: TestClient) -> None:
    resp = lexy_client.patch("/api/v1/persona", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "no-op"


# ─── Agent integration ─────────────────────────────────────────────────────


def test_agent_plan_uses_persona_prompt(lexy_client: TestClient) -> None:
    """Verify the agent builds its system message from the current persona."""
    app = lexy_client.app.state.lexy

    saved_persona = app.persona

    app.persona = Persona(
        name="Marker",
        user_name="TestUser",
        sections=PersonaSections(
            identity="MARKER_IDENTITY_12345",
        ),
    )

    async def run_plan() -> list[dict[str, str]]:
        ctx = {
            "text": "hi",
            "session_id": "test",
            "user_id": "test",
            "brain": "auto",
            "tools_used": [],
            "recalled": [],
        }
        return await app.agent._plan(ctx)

    messages = asyncio.get_event_loop().run_until_complete(run_plan())
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "MARKER_IDENTITY_12345" in system_msg["content"]
    # Protected sections should also be present
    assert "## Wie du denkst" in system_msg["content"]

    app.persona = saved_persona
