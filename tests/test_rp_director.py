"""Tests for the rp_director plugin — state machine and prompt assembly.

Plugin-level tests for the Director plugin focus on the two pure
components that don't need a running ``LexyApp``:

* :class:`DirectorState` — SQLite state transitions for in-flight setups.
* :func:`assemble_director_prompt` / :func:`render_state_block` — the
  prompt-builder that the ``before_prompt_build`` hook hands to the LLM.

Integration of the hook + tools with character_chat is covered by manual
end-to-end verification (see plan file). Mocking the full agent loop +
tool registry here would mostly test the mocks.
"""

from __future__ import annotations

import aiosqlite
import pytest

from plugins.rp_director.director_state import DirectorState
from plugins.rp_director.prompts import (
    TOOL_GUIDE,
    assemble_director_prompt,
    render_state_block,
)


# ─── DirectorState ──────────────────────────────────────────────────────


class TestDirectorState:
    @pytest.mark.asyncio
    async def test_start_creates_collecting_session(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        record = await state.start("sess-1", user_intent="space horror RP")

        assert record["session_id"] == "sess-1"
        assert record["state"] == "collecting"
        assert record["user_intent"] == "space horror RP"
        assert record["scenario"] is None
        assert record["characters"] == []
        await db.close()

    @pytest.mark.asyncio
    async def test_is_active_only_for_in_flight_states(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        assert await state.is_active("missing") is False

        await state.start("s1")
        assert await state.is_active("s1") is True

        await state.mark_committed("s1")
        assert await state.is_active("s1") is False

        await state.start("s2")
        await state.mark_cancelled("s2")
        assert await state.is_active("s2") is False
        await db.close()

    @pytest.mark.asyncio
    async def test_set_scenario_transitions_to_proposing(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("s1", user_intent="dark fantasy")
        record = await state.set_scenario(
            "s1",
            {
                "setting": "Cursed forest",
                "mood": "Eerie",
                "hook": "A child gone missing",
                "scene_text": "Du stehst am Waldrand, der Nebel zieht auf.",
            },
        )

        assert record is not None
        assert record["state"] == "proposing"
        assert record["scenario"]["setting"] == "Cursed forest"
        assert "scene_text" in record["scenario"]
        await db.close()

    @pytest.mark.asyncio
    async def test_set_characters_replaces_full_list(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("s1")
        await state.set_characters(
            "s1",
            [{"name": "Alice"}, {"name": "Bob"}],
        )
        # Second call replaces, not appends.
        record = await state.set_characters("s1", [{"name": "Cara"}])

        assert record is not None
        assert [c["name"] for c in record["characters"]] == ["Cara"]
        await db.close()

    @pytest.mark.asyncio
    async def test_committed_session_rejects_further_proposals(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("s1")
        await state.set_scenario("s1", {"setting": "X"})
        await state.mark_committed("s1")

        # set_scenario filters on state — a committed session should not
        # have its scenario field overwritten.
        result = await state.set_scenario("s1", {"setting": "Y"})
        assert result is not None
        assert result["state"] == "committed"
        assert result["scenario"]["setting"] == "X"
        await db.close()

    @pytest.mark.asyncio
    async def test_list_active_excludes_terminal(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("a")
        await state.start("b")
        await state.start("c")
        await state.mark_committed("a")
        await state.mark_cancelled("b")

        active = await state.list_active()
        assert [r["session_id"] for r in active] == ["c"]
        await db.close()

    @pytest.mark.asyncio
    async def test_expire_idle_disabled_when_zero(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("s1")
        expired = await state.expire_idle(0)
        assert expired == []
        assert await state.is_active("s1") is True
        await db.close()

    @pytest.mark.asyncio
    async def test_expire_idle_cancels_old_sessions(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("fresh")
        await state.start("stale")
        # Manually backdate "stale" so expire_idle picks it up.
        await db.execute(
            "UPDATE rp_director_sessions SET updated_at = 0 WHERE session_id = 'stale'"
        )
        await db.commit()

        expired = await state.expire_idle(60.0)
        assert expired == ["stale"]
        assert await state.is_active("fresh") is True
        assert await state.is_active("stale") is False
        await db.close()

    @pytest.mark.asyncio
    async def test_corrupt_json_yields_empty_lists(self) -> None:
        db = await aiosqlite.connect(":memory:")
        state = DirectorState(db)
        await state.init_table()

        await state.start("s1")
        # Corrupt characters_json directly.
        await db.execute(
            "UPDATE rp_director_sessions SET characters_json = '{not json' "
            "WHERE session_id = 's1'"
        )
        await db.commit()

        record = await state.get("s1")
        assert record is not None
        assert record["characters"] == []
        await db.close()


# ─── Prompt assembly ────────────────────────────────────────────────────


class TestRenderStateBlock:
    def test_empty_record_shows_placeholders(self) -> None:
        record = {
            "scenario": None,
            "characters": [],
            "user_intent": "",
        }
        block = render_state_block(record)
        assert "Scenario: _noch nicht vorgeschlagen_" in block
        assert "Charaktere: _noch keine vorgeschlagen_" in block

    def test_user_intent_rendered(self) -> None:
        block = render_state_block(
            {
                "scenario": None,
                "characters": [],
                "user_intent": "Heist im Cyberpunk-Setting",
            }
        )
        assert "Heist im Cyberpunk-Setting" in block

    def test_scenario_fields_rendered(self) -> None:
        block = render_state_block(
            {
                "scenario": {
                    "setting": "Nachtclub",
                    "mood": "düster",
                    "hook": "Diebstahl eines Datacubes",
                },
                "characters": [],
                "user_intent": "",
            }
        )
        assert "Setting: Nachtclub" in block
        assert "Stimmung: düster" in block
        assert "Plot-Hook: Diebstahl eines Datacubes" in block

    def test_characters_with_relationships(self) -> None:
        block = render_state_block(
            {
                "scenario": None,
                "characters": [
                    {
                        "name": "Iko",
                        "age_stage": "teen",
                        "persona": "Neugierig, glaubt nicht an Geister.",
                        "relationships": {"Mara": "Mentorin"},
                    },
                ],
                "user_intent": "",
            }
        )
        assert "Iko (teen)" in block
        assert "Mara: Mentorin" in block

    def test_proposed_label_warns_not_yet_committed(self) -> None:
        block = render_state_block(
            {
                "scenario": {"setting": "X"},
                "characters": [],
                "user_intent": "",
            }
        )
        assert "noch nicht committed" in block

    def test_autonomy_undecided_flagged_as_required(self) -> None:
        # When the Director has a scenario but no autonomy mode picked yet,
        # the state block must remind the LLM that this is a hard
        # prerequisite for commit. Otherwise it tends to skip the question.
        block = render_state_block(
            {
                "scenario": {"setting": "X"},
                "characters": [],
                "user_intent": "",
            }
        )
        assert "Autonomie: _noch nicht entschieden_" in block
        assert "PFLICHT vor commit" in block

    def test_autonomy_proactive_with_interval_rendered(self) -> None:
        block = render_state_block(
            {
                "scenario": {
                    "setting": "X",
                    "autonomy": {"mode": "proactive", "pulse_minutes": 20},
                },
                "characters": [],
                "user_intent": "",
            }
        )
        assert "Autonomie: proactive" in block
        assert "alle 20min pro Char" in block

    def test_autonomy_simulation_with_char_mode_rendered(self) -> None:
        block = render_state_block(
            {
                "scenario": {
                    "setting": "X",
                    "autonomy": {
                        "mode": "simulation",
                        "simulation_interval_minutes": 5,
                        "character_mode": 2,
                    },
                },
                "characters": [],
                "user_intent": "",
            }
        )
        assert "Autonomie: simulation" in block
        assert "alle 5min ein zufaelliger Char" in block
        assert "character_mode=2" in block


class TestAssembleDirectorPrompt:
    def test_combines_persona_and_state(self) -> None:
        prompt = assemble_director_prompt(
            persona_prompt="## Wer du bist\nDu bist der Director.",
            state_record={
                "scenario": None,
                "characters": [],
                "user_intent": "Test",
            },
            user_name="Mike",
        )
        assert "Du bist der Director." in prompt
        # Tool guide always present.
        assert "propose_scenario" in prompt
        assert "propose_characters" in prompt
        assert "commit_rp_setup" in prompt
        # State block present.
        assert "Aktueller Setup-Stand" in prompt

    def test_user_name_substituted_in_tool_guide(self) -> None:
        prompt = assemble_director_prompt(
            persona_prompt="(persona)",
            state_record={"scenario": None, "characters": [], "user_intent": ""},
            user_name="Alex",
        )
        assert "Alex" in prompt
        # Make sure the placeholder was actually replaced.
        assert "{user_name}" not in prompt

    def test_tool_guide_lists_all_five_tools(self) -> None:
        # Sanity check on the source of truth — if a tool is added/renamed
        # in the plugin, this test catches the doc drift.
        for tool in (
            "propose_scenario",
            "propose_characters",
            "commit_rp_setup",
            "cancel_rp_setup",
            "set_rp_autonomy",
        ):
            assert tool in TOOL_GUIDE

    def test_tool_guide_documents_autonomy_modes(self) -> None:
        # The Director must learn when to use which mode — if these strings
        # disappear from the guide the LLM stops asking the right question.
        for keyword in (
            "addressed_only",
            "proactive",
            "simulation",
            "pulse_minutes",
            "simulation_interval_minutes",
            "character_mode",
        ):
            assert keyword in TOOL_GUIDE
