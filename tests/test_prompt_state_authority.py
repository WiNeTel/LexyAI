"""
Hotfix tests — prompt's state block is the authoritative source.

Mike reported: "Sandra verweist auf ein Shirt das sie laut Charakter
nicht an hat". Diagnosis: when ``state.clothing="nackt"`` was set,
the system prompt had:

  ## Dein Zustand
  **Kleidung:** nackt

  ## Beispiel-Dialog
  *Sandra zupft an ihrem Shirt zurecht*  ← stale!
  ...

The LLM read both sources and treated example_dialog as a few-shot
training signal — and emitted Shirt-references in its turn. Plus
the older RP-discipline rules only listed three keys for
``<state>`` (location, mood, last_action), so the LLM literally
couldn't update its own clothing memory.

This file pins three things going forward:

1. The "## Dein Zustand" header is now ``## Dein Zustand (AKTUELL …)``
   with an explicit warning that beats the example_dialog.
2. The example_dialog is labelled "NUR Stilreferenz, NICHT aktuell".
3. The ``<state>`` instruction lists all six anchor keys + mentions
   free-form snake_case keys.

Plus we test the new prompt-preview REST endpoint that lets Mike
inspect exactly what's being sent to the LLM.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from lexy_core.app import LexyApp
from plugins.character_chat.character_card import CharacterCard


# ─── Direct prompt-builder tests (no HTTP) ───────────────────────────


class TestPromptStateAuthority:
    def test_state_block_has_authority_warning(self) -> None:
        card = CharacterCard(
            name="Sandra",
            persona="Sandra ist 13 Jahre alt.",
            example_dialog="*zupft an ihrem Shirt zurecht*",
            state={"clothing": "nackt"},
        )
        prompt = card.build_system_prompt()
        # Header signals "this is the truth".
        assert "## Dein Zustand (AKTUELL" in prompt
        # Explicit warning that overrides the example_dialog.
        assert "WICHTIG" in prompt
        assert "ist DAS NUR STILREFERENZ" in prompt
        # State value renders.
        assert "Kleidung:** nackt" in prompt

    def test_example_dialog_section_is_marked_as_stylereference(self) -> None:
        card = CharacterCard(
            name="Lena",
            persona="Lena ist 6.",
            example_dialog="*lacht* Hi!",
            state={"clothing": "nackt"},
        )
        prompt = card.build_system_prompt()
        assert "## Beispiel-Dialog (NUR Stilreferenz, NICHT aktuell!)" in prompt

    def test_clothes_rule_in_rp_disziplin(self) -> None:
        """The new rule explicitly bans hallucinated clothes."""
        card = CharacterCard(
            name="Sandra", persona="x", state={"clothing": "nackt"},
        )
        prompt = card.build_system_prompt()
        assert "Klamotten + Körper" in prompt
        assert "KEINE Halluzinationen" in prompt

    def test_state_block_lists_all_six_anchor_keys(self) -> None:
        card = CharacterCard(name="Sandra", persona="x")
        prompt = card.build_system_prompt()
        # All six anchor keys must be allowed in <state>.
        for key in (
            "location", "mood", "last_action",
            "clothing", "posture", "condition",
        ):
            assert key in prompt, f"<state> instruction missing key {key!r}"

    def test_no_state_block_when_state_empty(self) -> None:
        """If state is empty we should NOT render the new "AKTUELL"
        section — otherwise we'd lecture the LLM about a non-existent
        truth."""
        card = CharacterCard(name="Sandra", persona="x", state={})
        prompt = card.build_system_prompt()
        assert "## Dein Zustand (AKTUELL" not in prompt
        # But the <state> instruction at the bottom is always there
        # (the LLM still might want to start tracking).
        assert "<state>" in prompt

    def test_state_block_renders_before_example_dialog(self) -> None:
        """Ordering matters: state authority is at the top, example
        dialog below. Otherwise the few-shot wins by recency bias."""
        card = CharacterCard(
            name="Sandra",
            persona="x",
            example_dialog="*style*",
            state={"clothing": "nackt"},
        )
        prompt = card.build_system_prompt()
        state_pos = prompt.find("## Dein Zustand")
        example_pos = prompt.find("## Beispiel-Dialog")
        assert state_pos > 0
        assert example_pos > state_pos


# ─── REST debug endpoint ────────────────────────────────────────────


@pytest.fixture(scope="module")
def lexy_client() -> TestClient:
    app = LexyApp("config/config.yaml")
    asyncio.get_event_loop().run_until_complete(app.startup())
    client = TestClient(app.fastapi)
    yield client
    asyncio.get_event_loop().run_until_complete(app.shutdown())


def _create_test_character(client: TestClient, **fields: object) -> str:
    """Spawn a character via the WS-mirror REST? Actually, the only
    way to create a character today is via WS, which the TestClient
    can't drive synchronously. So we go straight at the store."""
    plugin = client.app.state.lexy.plugin_loader.get_plugin("character_chat")  # type: ignore[attr-defined]
    store = plugin._store

    payload = {
        "name": fields.pop("name", f"Test-{uuid.uuid4().hex[:6]}"),
        "persona": fields.pop("persona", "test persona"),
        "greeting": fields.pop("greeting", "hi"),
        "example_dialog": fields.pop("example_dialog", "*test*"),
    }
    payload.update(fields)
    card = CharacterCard(**payload)
    created = asyncio.get_event_loop().run_until_complete(store.create(card))
    return created.id


def test_prompt_preview_returns_rendered_prompt(
    lexy_client: TestClient,
) -> None:
    char_id = _create_test_character(
        lexy_client,
        name=f"PromptDebug-{uuid.uuid4().hex[:6]}",
        persona="Sandra ist 13.",
        example_dialog="*zupft an ihrem Shirt*",
        state={"clothing": "nackt"},
    )
    resp = lexy_client.get(
        f"/api/v1/plugins/character_chat/characters/{char_id}/prompt"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["character_id"] == char_id
    assert "Kleidung:** nackt" in body["prompt"]
    assert body["state"] == {"clothing": "nackt"}
    # Diagnostic fields surface so Mike can see WHICH source has
    # the stale text:
    assert body["persona"] == "Sandra ist 13."
    assert "Shirt" in body["example_dialog"]


def test_prompt_preview_unknown_character_returns_404(
    lexy_client: TestClient,
) -> None:
    resp = lexy_client.get(
        "/api/v1/plugins/character_chat/characters/__nope__/prompt"
    )
    assert resp.status_code == 404


def test_prompt_preview_with_session_id_includes_other_characters(
    lexy_client: TestClient,
) -> None:
    """When ``session_id`` is given and other characters are bound to
    that session, they show up under '## Andere Anwesende'."""
    plugin = lexy_client.app.state.lexy.plugin_loader.get_plugin("character_chat")  # type: ignore[attr-defined]
    store = plugin._store

    sid = f"prompt-debug-sess-{uuid.uuid4().hex[:6]}"
    a_id = _create_test_character(
        lexy_client, name=f"Alpha-{uuid.uuid4().hex[:6]}", persona="alpha p",
    )
    b_id = _create_test_character(
        lexy_client, name=f"Beta-{uuid.uuid4().hex[:6]}", persona="beta p",
    )
    asyncio.get_event_loop().run_until_complete(
        store.attach_to_session(a_id, sid)
    )
    asyncio.get_event_loop().run_until_complete(
        store.attach_to_session(b_id, sid)
    )

    resp = lexy_client.get(
        f"/api/v1/plugins/character_chat/characters/{a_id}/prompt"
        f"?session_id={sid}"
    )
    assert resp.status_code == 200
    body = resp.json()
    # Alpha's prompt should mention Beta in "Andere Anwesende".
    assert "## Andere Anwesende" in body["prompt"]
    assert "Beta-" in body["prompt"]
    # And NOT Alpha herself (she's the speaker, not a peer).
    assert body["prompt"].count("Du bist Alpha") <= 2  # in header + maybe persona
