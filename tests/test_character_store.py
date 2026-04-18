"""
Tests for the character_chat plugin's CharacterStore + Silly-Tavern import.

We instantiate the store against an in-memory aiosqlite connection so the
tests don't pollute ``data/plugins/character_chat/`` and run fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from plugins.character_chat.character_card import (
    AGE_STAGES,
    CharacterCard,
    CharacterCardError,
    parse_silly_tavern_card,
    parse_silly_tavern_file,
)
from plugins.character_chat.character_store import CharacterStore


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> CharacterStore:
    db = await aiosqlite.connect(":memory:")
    s = CharacterStore(db)
    await s.init_schema()
    try:
        yield s
    finally:
        await db.close()


# ─── CharacterCard model ─────────────────────────────────────────────────────


def test_character_card_requires_name() -> None:
    # Pydantic wraps validator errors in ValidationError; we accept any
    # exception — the contract is "construction fails", not the concrete
    # class (project_store uses the same pattern).
    with pytest.raises(Exception):
        CharacterCard(name="   ")


def test_character_card_rejects_bad_age_stage() -> None:
    with pytest.raises(Exception):
        CharacterCard(name="Luna", age_stage="ancient")


def test_character_card_rejects_bad_color() -> None:
    with pytest.raises(Exception):
        CharacterCard(name="Luna", color="mauve")


def test_character_card_defaults_color_when_empty() -> None:
    card = CharacterCard(name="Luna", color="")
    assert card.color == "#7aa2f7"


def test_character_card_to_row_round_trip() -> None:
    card = CharacterCard(
        name="Luna",
        persona="Ein Säugling mit leuchtenden Augen.",
        age_stage="baby",
        color="#ff88cc",
        relationships={"lexy123": "Mutter"},
        tags=["rp", "family"],
        active_sessions=["sess-42"],
        proactive_pulse_pattern="every 3h",
    )
    row = card.to_row()
    # Round-trip through JSON since that's what SQLite gives us back.
    row["relationships"] = json.loads(row["relationships"])
    row["tags"] = json.loads(row["tags"])
    row["active_sessions"] = json.loads(row["active_sessions"])
    rebuilt = CharacterCard.from_row(
        {
            **row,
            "relationships": json.dumps(row["relationships"]),
            "tags": json.dumps(row["tags"]),
            "active_sessions": json.dumps(row["active_sessions"]),
        }
    )
    assert rebuilt.name == "Luna"
    assert rebuilt.age_stage == "baby"
    assert rebuilt.relationships == {"lexy123": "Mutter"}
    assert rebuilt.tags == ["rp", "family"]
    assert rebuilt.active_sessions == ["sess-42"]


def test_build_system_prompt_contains_persona_and_scenario() -> None:
    card = CharacterCard(
        name="Luna",
        persona="Ein wildes Kind mit Sommersprossen.",
        scenario="Es ist Weihnachten am Kamin.",
        example_dialog="Luna: Mama, kann ich aufbleiben?",
    )
    prompt = card.build_system_prompt()
    assert "Du bist Luna" in prompt
    assert "wildes Kind" in prompt
    assert "Weihnachten am Kamin" in prompt
    assert "Mama, kann ich aufbleiben" in prompt
    assert "[PASS]" not in prompt  # [PASS] is appended by orchestrator, not the card


def test_build_system_prompt_renders_other_characters_with_relationships() -> None:
    lexy = CharacterCard(id="lexy", name="Lexy", persona="KI-Assistentin")
    luna = CharacterCard(
        id="luna",
        name="Luna",
        persona="Tochter",
        relationships={"lexy": "Mama"},
    )
    prompt = luna.build_system_prompt(other_characters=[lexy, luna])
    assert "Andere Anwesende" in prompt
    assert "Lexy: Mama" in prompt
    # Luna shouldn't include herself in the roster.
    roster_block = prompt.split("## Andere Anwesende", 1)[1]
    assert "Luna" not in roster_block.split("##", 1)[0].replace("Lexy", "")


def test_build_system_prompt_adds_age_stage_guidance_for_baby() -> None:
    card = CharacterCard(name="Luna", age_stage="baby")
    prompt = card.build_system_prompt()
    assert "Säugling" in prompt
    assert "*Aktionen*" in prompt or "*" in prompt


# ─── Store CRUD ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_and_get(store: CharacterStore) -> None:
    card = CharacterCard(name="Luna", persona="Baby")
    await store.create(card)
    loaded = await store.get(card.id)
    assert loaded is not None
    assert loaded.name == "Luna"
    assert loaded.persona == "Baby"


@pytest.mark.asyncio
async def test_create_rejects_name_collision(store: CharacterStore) -> None:
    await store.create(CharacterCard(name="Luna"))
    with pytest.raises(CharacterCardError):
        await store.create(CharacterCard(name="Luna"))


@pytest.mark.asyncio
async def test_create_allows_name_reuse_after_archive(
    store: CharacterStore,
) -> None:
    first = await store.create(CharacterCard(name="Luna"))
    await store.archive(first.id)
    # Archiving frees the name — Decision #4 implies we isolate memory per id,
    # so a new Luna is genuinely a new character.
    second = await store.create(CharacterCard(name="Luna"))
    assert second.id != first.id


@pytest.mark.asyncio
async def test_list_excludes_archived_by_default(store: CharacterStore) -> None:
    a = await store.create(CharacterCard(name="Alice"))
    b = await store.create(CharacterCard(name="Bob"))
    await store.archive(a.id)
    cards = await store.list()
    assert [c.name for c in cards] == ["Bob"]
    all_cards = await store.list(include_archived=True)
    assert {c.name for c in all_cards} == {"Alice", "Bob"}


@pytest.mark.asyncio
async def test_update_fields(store: CharacterStore) -> None:
    card = await store.create(CharacterCard(name="Luna", age_stage="baby"))
    updated = await store.update(
        card.id,
        persona="Kleines Mädchen mit Locken.",
        age_stage="toddler",
        color="#ff00ff",
    )
    assert updated is not None
    assert updated.persona == "Kleines Mädchen mit Locken."
    assert updated.age_stage == "toddler"
    assert updated.color == "#ff00ff"


@pytest.mark.asyncio
async def test_update_unknown_returns_none(store: CharacterStore) -> None:
    assert await store.update("does-not-exist", name="X") is None


@pytest.mark.asyncio
async def test_update_rejects_invalid_age_stage(store: CharacterStore) -> None:
    card = await store.create(CharacterCard(name="Luna"))
    with pytest.raises(CharacterCardError):
        await store.update(card.id, age_stage="ancient")


@pytest.mark.asyncio
async def test_update_rejects_name_collision(store: CharacterStore) -> None:
    a = await store.create(CharacterCard(name="Alice"))
    await store.create(CharacterCard(name="Bob"))
    with pytest.raises(CharacterCardError):
        await store.update(a.id, name="Bob")


@pytest.mark.asyncio
async def test_delete_removes_card(store: CharacterStore) -> None:
    card = await store.create(CharacterCard(name="Luna"))
    assert await store.delete(card.id) is True
    assert await store.get(card.id) is None
    assert await store.delete(card.id) is False


@pytest.mark.asyncio
async def test_archive_then_unarchive(store: CharacterStore) -> None:
    card = await store.create(CharacterCard(name="Luna"))
    assert await store.archive(card.id) is True
    archived = await store.get(card.id)
    assert archived is not None and archived.archived is True
    assert await store.unarchive(card.id) is True
    alive = await store.get(card.id)
    assert alive is not None and alive.archived is False


# ─── Session membership ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attach_and_list_in_session(store: CharacterStore) -> None:
    luna = await store.create(CharacterCard(name="Luna"))
    lexy = await store.create(CharacterCard(name="Lexy"))
    unrelated = await store.create(CharacterCard(name="Bob"))
    await store.attach_to_session(luna.id, "sess-1")
    await store.attach_to_session(lexy.id, "sess-1")
    await store.attach_to_session(unrelated.id, "sess-2")
    in_sess1 = await store.list_in_session("sess-1")
    assert {c.name for c in in_sess1} == {"Luna", "Lexy"}


@pytest.mark.asyncio
async def test_attach_is_idempotent(store: CharacterStore) -> None:
    luna = await store.create(CharacterCard(name="Luna"))
    await store.attach_to_session(luna.id, "sess-1")
    await store.attach_to_session(luna.id, "sess-1")
    card = await store.get(luna.id)
    assert card is not None
    assert card.active_sessions == ["sess-1"]


@pytest.mark.asyncio
async def test_detach_removes_session_membership(store: CharacterStore) -> None:
    luna = await store.create(CharacterCard(name="Luna"))
    await store.attach_to_session(luna.id, "sess-1")
    await store.attach_to_session(luna.id, "sess-2")
    await store.detach_from_session(luna.id, "sess-1")
    card = await store.get(luna.id)
    assert card is not None
    assert card.active_sessions == ["sess-2"]


@pytest.mark.asyncio
async def test_archived_characters_excluded_from_session_list(
    store: CharacterStore,
) -> None:
    luna = await store.create(CharacterCard(name="Luna"))
    await store.attach_to_session(luna.id, "sess-1")
    await store.archive(luna.id)
    # list_in_session defaults to include_archived=False.
    assert await store.list_in_session("sess-1") == []
    visible = await store.list_in_session("sess-1", include_archived=True)
    assert [c.name for c in visible] == ["Luna"]


# ─── Silly-Tavern import ────────────────────────────────────────────────────


_V1_CARD: dict = {
    "name": "Aria",
    "description": "Eine mysteriöse Bibliothekarin mit einem Geheimnis.",
    "scenario": "In einer verschlossenen Bibliothek bei Nacht.",
    "first_mes": "Du hast dich verlaufen. Soll ich dir helfen?",
    "mes_example": "<START>\nAria: Komm her, ich zeige dir das Buch.",
    "tags": ["mystery, library, female"],
}


_V2_CARD: dict = {
    "spec": "chara_card_v2",
    "data": {
        "name": "Kai",
        "description": "Ein ruhiger Samurai mit schnellen Reflexen.",
        "personality": "stoisch, loyal, wortkarg",
        "scenario": "Am Rande eines Bambuswaldes im Mondschein.",
        "first_mes": "Du solltest hier nicht sein, Fremder.",
        "mes_example": "Kai: *zieht das Schwert halb aus der Scheide*",
        "tags": ["samurai", "male", "action"],
    },
}


def test_parse_v1_card_maps_description_to_persona() -> None:
    card = parse_silly_tavern_card(_V1_CARD)
    assert card.name == "Aria"
    assert "mysteriöse Bibliothekarin" in card.persona
    assert card.scenario.startswith("In einer verschlossenen")
    assert card.greeting.startswith("Du hast dich verlaufen")
    assert "Komm her" in card.example_dialog
    # Comma-separated tag strings are split.
    assert set(card.tags) == {"mystery", "library", "female"}


def test_parse_v2_card_reads_data_block_and_personality() -> None:
    card = parse_silly_tavern_card(_V2_CARD)
    assert card.name == "Kai"
    # v2 can put the persona under "personality" as well; we pick whichever
    # is non-empty (description has priority).
    assert "Samurai" in card.persona
    assert card.scenario.startswith("Am Rande")
    assert card.greeting.startswith("Du solltest")
    assert set(card.tags) == {"samurai", "male", "action"}


def test_parse_invalid_payload_raises() -> None:
    with pytest.raises(CharacterCardError):
        parse_silly_tavern_card({"description": "no name here"})
    with pytest.raises(CharacterCardError):
        parse_silly_tavern_card("just a string")  # type: ignore[arg-type]


def test_parse_silly_tavern_file_reads_json(tmp_path: Path) -> None:
    p = tmp_path / "aria.json"
    p.write_text(json.dumps(_V1_CARD), encoding="utf-8")
    card = parse_silly_tavern_file(p)
    assert card.name == "Aria"


def test_parse_silly_tavern_file_bad_path_raises(tmp_path: Path) -> None:
    with pytest.raises(CharacterCardError):
        parse_silly_tavern_file(tmp_path / "missing.json")


def test_parse_silly_tavern_file_bad_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CharacterCardError):
        parse_silly_tavern_file(p)


@pytest.mark.asyncio
async def test_store_import_silly_tavern_persists_card(
    store: CharacterStore,
) -> None:
    card = await store.import_silly_tavern(
        _V2_CARD, color="#ff5588", age_stage="adult"
    )
    assert card.name == "Kai"
    assert card.color == "#ff5588"
    from_db = await store.get(card.id)
    assert from_db is not None
    assert from_db.name == "Kai"


# ─── Bulk insert helper ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bulk_insert_skips_collisions(store: CharacterStore) -> None:
    cards = [
        CharacterCard(name="Luna"),
        CharacterCard(name="Lexy"),
        CharacterCard(name="Luna"),  # collision with first
    ]
    inserted = await store.bulk_insert(cards)
    assert inserted == 2
    count = await store.count()
    assert count == 2


# ─── Sanity: AGE_STAGES constant ────────────────────────────────────────────


def test_age_stages_cover_full_ladder() -> None:
    assert AGE_STAGES == ("baby", "toddler", "child", "teen", "adult")
