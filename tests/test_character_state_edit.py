"""
Hotfix tests — character state can be edited via the update path.

Mike reported: "Ich habe die Charaktere geändert, das sie jetzt Nackt
dein sollten! ... aber Sandra verweist auf ein Shirt das sie nicht an
hat". Diagnosis: the character form had no UI control for ``state``
— it only edited persona/scenario. The ``state`` dict (clothing,
posture, condition, mood, location, plus free-form keys the LLM
adds via ``<state>...</state>`` blocks) lived only in the DB and
mutated only via the LLM's emitted state blocks. Mike couldn't
manually clear "clothing: Shirt" so Sandra kept "wearing" it.

Fix is mostly UI (HTML + JS), but we test the backend path the UI
calls into:

* ``CharacterStore.update(state={...})`` writes a new state JSON
  on the row + the next ``get`` returns the merged card.
* Empty state dict (the form's "Reset" + Save outcome) clears the
  state column to ``{}``.
* Empty-string anchor values follow the ``merge_state`` semantics:
  on a TURN-driven update they'd clear the key; the direct
  ``update()`` call replaces the whole dict so an empty input maps
  to "drop the key entirely".

The tests use an in-memory aiosqlite DB so there's no LexyApp boot
overhead.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
import pytest
import pytest_asyncio

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.character_store import CharacterStore


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> CharacterStore:
    db = await aiosqlite.connect(":memory:")
    s = CharacterStore(db)
    await s.init_schema()
    yield s
    await db.close()


async def _spawn(store: CharacterStore, **overrides: Any) -> CharacterCard:
    """Build a CharacterCard with sensible defaults and persist it."""
    fields = {
        "name": overrides.pop("name", "Sandra"),
        "persona": overrides.pop("persona", "An ordinary teen."),
        "greeting": overrides.pop("greeting", "Hi."),
    }
    fields.update(overrides)
    card = CharacterCard(**fields)
    return await store.create(card)


# ─── Direct-update tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_can_be_replaced_via_update(
    store: CharacterStore,
) -> None:
    """The form's submitCharacterForm sends ``state: {clothing:
    'nackt', ...}`` and expects that to overwrite the row's state."""
    card = await _spawn(store, name="StateUpdateTest")
    # Seed a stale state — simulating the LLM's previous "Shirt" entry.
    await store.update(card.id, state={"clothing": "Shirt", "mood": "happy"})

    # Mike clears clothing via the form; he keeps mood = "happy".
    await store.update(
        card.id, state={"mood": "happy"}
    )

    refreshed = await store.get(card.id)
    assert refreshed is not None
    assert refreshed.state == {"mood": "happy"}
    assert "clothing" not in refreshed.state


@pytest.mark.asyncio
async def test_state_can_be_fully_emptied_via_update(
    store: CharacterStore,
) -> None:
    """Form's "Reset State" button + Save sends ``state: {}``. The
    row's state column must end up empty after that."""
    card = await _spawn(store, name="StateEmptyTest")
    await store.update(
        card.id,
        state={"clothing": "Shirt", "posture": "sitzend", "mood": "müde"},
    )
    # Sanity: state was set.
    pre = await store.get(card.id)
    assert pre is not None
    assert pre.state == {
        "clothing": "Shirt", "posture": "sitzend", "mood": "müde",
    }

    # Reset.
    await store.update(card.id, state={})

    refreshed = await store.get(card.id)
    assert refreshed is not None
    assert refreshed.state == {}


@pytest.mark.asyncio
async def test_update_state_does_not_touch_other_fields(
    store: CharacterStore,
) -> None:
    """Editing state shouldn't reset persona/scenario/etc.

    Defends against a refactor that would accidentally pass too
    many fields through the merge.
    """
    card = await _spawn(
        store, name="StateIsolation", persona="Original persona text",
    )
    await store.update(card.id, state={"clothing": "nackt"})

    refreshed = await store.get(card.id)
    assert refreshed is not None
    assert refreshed.persona == "Original persona text"
    assert refreshed.state == {"clothing": "nackt"}


@pytest.mark.asyncio
async def test_freeform_state_keys_round_trip(
    store: CharacterStore,
) -> None:
    """Anchor + free-form keys both round-trip through the JSON column."""
    card = await _spawn(store, name="FreeformState")
    await store.update(
        card.id,
        state={
            "clothing": "nackt",
            "wet": "true",  # free-form snake_case key
            "energy_level": "high",
        },
    )
    refreshed = await store.get(card.id)
    assert refreshed is not None
    assert refreshed.state["clothing"] == "nackt"
    assert refreshed.state["wet"] == "true"
    assert refreshed.state["energy_level"] == "high"
