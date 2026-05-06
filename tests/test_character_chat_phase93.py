"""
Tests for the Phase-9.3 character_chat upgrades:

* NL-mention parser (no @-prefix needed)
* Character state field + state-block parser/merger
* Pulse generator (LLM-driven proactive pulses)
* Pulse-mention propagation (Char A pulses, Char B answers in same round)

All tests use deterministic fakes — no real LLM, no real DB unless an
in-memory aiosqlite connection makes the test more realistic than mocks.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
import pytest

from plugins.character_chat.character_card import (
    CharacterCard,
    _format_state_block,
)
from plugins.character_chat.character_store import CharacterStore
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
)
from plugins.character_chat.mention_parser import parse_nl_mentions
from plugins.character_chat.pulse_generator import PulseGenerator
from plugins.character_chat.state_updater import (
    KNOWN_STATE_KEYS,
    merge_state,
    parse_state_block,
)


# ─── Phase A — NL-mention parser ─────────────────────────────────────────


class TestNLMentionParser:
    def setup_method(self) -> None:
        self.mara = CharacterCard(id="m", name="Mara")
        self.drell = CharacterCard(id="d", name="Drell")
        self.iko = CharacterCard(id="i", name="Iko")

    def test_empty_text_returns_empty(self) -> None:
        assert parse_nl_mentions("", [self.mara, self.drell]) == []

    def test_no_candidates_returns_empty(self) -> None:
        assert parse_nl_mentions("Mara, hilf mir", []) == []

    def test_single_mention(self) -> None:
        assert parse_nl_mentions("Mara, hilf mir", [self.mara, self.drell]) == ["m"]

    def test_two_mentions_in_order(self) -> None:
        text = "Mara, schau mal zum Captain. Drell, du sicherst die Tür!"
        assert parse_nl_mentions(text, [self.mara, self.drell]) == ["m", "d"]

    def test_two_mentions_reverse_order(self) -> None:
        text = "Drell, geh los! Und Mara, du wartest hier."
        assert parse_nl_mentions(text, [self.mara, self.drell]) == ["d", "m"]

    def test_three_mentions_ordered_by_first_occurrence(self) -> None:
        text = "Iko vorne, Mara mittig, Drell hinten."
        result = parse_nl_mentions(text, [self.mara, self.drell, self.iko])
        assert result == ["i", "m", "d"]

    def test_word_boundary_no_partial_match(self) -> None:
        # "Maraschino" must NOT match "Mara".
        assert parse_nl_mentions("Maraschino-Kirschen", [self.mara]) == []

    def test_case_insensitive(self) -> None:
        assert parse_nl_mentions("MARA, gleich!", [self.mara]) == ["m"]
        assert parse_nl_mentions("mara, gleich!", [self.mara]) == ["m"]

    def test_punctuation_around_name(self) -> None:
        for punct in (",", ".", "?", "!", ":", ";", " "):
            text = f"Mara{punct}"
            assert parse_nl_mentions(text, [self.mara]) == ["m"], punct

    def test_duplicate_mentions_kept_at_first_position(self) -> None:
        # "Mara ... Mara" → still just ['m'] in the position of first occurrence.
        text = "Mara, geh. Mara, ich meine es ernst."
        assert parse_nl_mentions(text, [self.mara]) == ["m"]

    def test_archived_candidates_not_excluded_by_parser(self) -> None:
        # Filtering archived characters is the orchestrator's job, not the
        # parser's. The parser just reports name matches.
        archived = CharacterCard(id="a", name="Alva", archived=True)
        result = parse_nl_mentions("Alva, du auch", [archived])
        assert result == ["a"]


# ─── Phase B — state parser + merger ─────────────────────────────────────


class TestStateBlockParser:
    def test_no_block_returns_unchanged(self) -> None:
        text = "Eine ganz normale Antwort."
        cleaned, updates = parse_state_block(text)
        assert cleaned == text
        assert updates == {}

    def test_simple_block_extracted_and_stripped(self) -> None:
        text = (
            "*setzt sich auf den Sessel* Endlich Ruhe.\n"
            "<state>location=Wohnzimmer; mood=entspannt; "
            "last_action=Sich gesetzt</state>"
        )
        cleaned, updates = parse_state_block(text)
        assert "<state>" not in cleaned
        assert "Endlich Ruhe." in cleaned
        assert updates == {
            "location": "Wohnzimmer",
            "mood": "entspannt",
            "last_action": "Sich gesetzt",
        }

    def test_snake_case_freeform_keys_accepted(self) -> None:
        # Phase 9.5: state schema relaxed — snake_case keys are now
        # accepted as free-form additions to the anchor set so the LLM
        # can track clothing / posture / proximity / etc. The block
        # below has TWO valid keys (snake_case) and one anchor.
        text = "Hi.<state>secret_key=42; location=Bad; weather_cond=rainy</state>"
        cleaned, updates = parse_state_block(text)
        assert cleaned == "Hi."
        # All three are valid snake_case → all three kept.
        assert updates == {
            "secret_key": "42",
            "location": "Bad",
            "weather_cond": "rainy",
        }

    def test_non_snake_case_keys_still_rejected(self) -> None:
        # CamelCase / dots / spaces / hyphens never pass the validator
        # — only sane snake_case identifiers do.
        text = "<state>BadKey=x; with.dot=y; with space=z; valid_one=ok</state>"
        _, updates = parse_state_block(text)
        # ``BadKey`` is lowercased to ``badkey`` and accepted (snake_case);
        # the other malformed ones are dropped. Anchor-set membership is
        # not required since Phase 9.5.
        assert updates == {"badkey": "x", "valid_one": "ok"}

    def test_malformed_block_strips_but_returns_empty(self) -> None:
        text = "Hi.<state>this is garbage</state>"
        cleaned, updates = parse_state_block(text)
        assert cleaned == "Hi."
        assert updates == {}

    def test_multiple_blocks_merge_left_to_right(self) -> None:
        text = (
            "Erst.<state>location=A</state>"
            " Dann.<state>location=B; mood=froh</state>"
        )
        cleaned, updates = parse_state_block(text)
        assert "<state>" not in cleaned
        # Later block wins for ``location``.
        assert updates == {"location": "B", "mood": "froh"}

    def test_long_value_truncated(self) -> None:
        long_val = "x" * 300
        text = f"<state>location={long_val}</state>"
        _, updates = parse_state_block(text)
        assert updates["location"].endswith("…")
        assert len(updates["location"]) <= 121  # 120 + ellipsis

    def test_quoted_values_are_unquoted(self) -> None:
        text = '<state>location="Bad"; mood=\'müde\'</state>'
        _, updates = parse_state_block(text)
        assert updates == {"location": "Bad", "mood": "müde"}

    def test_known_keys_match_anchor_set(self) -> None:
        # Phase 9.5: KNOWN_STATE_KEYS is now an alias for the anchor set
        # (location, mood, last_action, clothing, posture, condition).
        # Free-form snake_case keys outside this set are accepted by
        # the parser/merger but don't get a localised label in the
        # prompt rendering — they're shown via TitleCase fallback.
        assert KNOWN_STATE_KEYS == frozenset(
            {
                "location", "mood", "last_action",
                "clothing", "posture", "condition",
            }
        )


class TestMergeState:
    def test_updates_overwrite_current(self) -> None:
        result = merge_state(
            {"location": "Bad", "mood": "müde"},
            {"mood": "wach"},
        )
        assert result == {"location": "Bad", "mood": "wach"}

    def test_empty_value_clears_field(self) -> None:
        result = merge_state(
            {"location": "Bad", "mood": "müde"},
            {"location": ""},
        )
        assert result == {"mood": "müde"}

    def test_freeform_snake_case_keys_accepted_in_merge(self) -> None:
        # Phase 9.5: snake_case keys are accepted, malformed ones
        # are still dropped — same validator as parse_state_block.
        result = merge_state(
            {"location": "Bad"},
            {"foo": "bar", "with space": "rejected"},
        )
        assert result == {"location": "Bad", "foo": "bar"}

    def test_empty_updates_returns_copy(self) -> None:
        current = {"location": "Bad"}
        result = merge_state(current, {})
        assert result == current
        assert result is not current  # defensive copy


# ─── Phase B — CharacterCard state field & SQLite round-trip ─────────────


class TestCharacterCardState:
    def test_default_state_is_empty_dict(self) -> None:
        c = CharacterCard(name="X")
        assert c.state == {}

    def test_state_renders_only_set_fields(self) -> None:
        rendered = _format_state_block({"location": "Bad", "mood": ""})
        assert "Bad" in rendered
        assert "Stimmung" not in rendered

    def test_state_render_empty_dict(self) -> None:
        assert _format_state_block({}) == ""

    def test_to_row_serialises_state(self) -> None:
        c = CharacterCard(name="X", state={"location": "Bad"})
        row = c.to_row()
        assert "state" in row
        assert "Bad" in row["state"]

    def test_from_row_round_trip(self) -> None:
        original = CharacterCard(
            name="X", state={"location": "Bad", "mood": "müde"}
        )
        round_tripped = CharacterCard.from_row(original.to_row())
        assert round_tripped.state == original.state


@pytest.mark.asyncio
async def test_store_persists_state() -> None:
    db = await aiosqlite.connect(":memory:")
    try:
        store = CharacterStore(db)
        await store.init_schema()
        card = CharacterCard(name="Mara", state={"location": "Brücke"})
        await store.create(card)
        loaded = await store.get(card.id)
        assert loaded is not None
        assert loaded.state == {"location": "Brücke"}

        # And the update path works on the new column.
        updated = await store.update(
            card.id,
            state={"location": "Brücke", "mood": "konzentriert"},
        )
        assert updated is not None
        assert updated.state == {"location": "Brücke", "mood": "konzentriert"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_store_migration_adds_state_column_on_pre_phase93_db() -> None:
    """Idempotent migration: an existing DB without the ``state`` column
    must get it on the next ``init_schema`` call without losing data.
    """
    db = await aiosqlite.connect(":memory:")
    try:
        # Simulate the pre-9.3 schema by recreating the table without ``state``.
        await db.execute(
            """
            CREATE TABLE characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                persona TEXT DEFAULT '',
                greeting TEXT DEFAULT '',
                scenario TEXT DEFAULT '',
                example_dialog TEXT DEFAULT '',
                avatar TEXT DEFAULT '',
                color TEXT DEFAULT '#7aa2f7',
                age_stage TEXT DEFAULT 'adult',
                voice TEXT DEFAULT '',
                relationships TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                active_sessions TEXT DEFAULT '[]',
                proactive_pulse_pattern TEXT DEFAULT '',
                proactive_pulse_prompt TEXT DEFAULT '',
                archived INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        await db.execute(
            "INSERT INTO characters (id, name, created_at, updated_at) "
            "VALUES ('legacy', 'Old Char', 0, 0)"
        )
        await db.commit()

        store = CharacterStore(db)
        await store.init_schema()

        # The migration should have added the column without dropping the row.
        cursor = await db.execute("PRAGMA table_info(characters)")
        cols = {r[1] for r in await cursor.fetchall()}
        await cursor.close()
        assert "state" in cols

        loaded = await store.get("legacy")
        assert loaded is not None
        assert loaded.name == "Old Char"
        assert loaded.state == {}
    finally:
        await db.close()


# ─── Phase C — Pulse generator (with fake LLM) ───────────────────────────


class _ScriptedLLM:
    """Minimal LLM stand-in that returns one scripted response per call."""

    def __init__(self, *, response: str = "", raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        messages: list[dict[str, str]],
        brain: str = "e4b",
        max_tokens: int = 200,
        temperature: float = 0.5,
        **extras: Any,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "brain": brain,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **extras,
            }
        )
        if self.raises is not None:
            raise self.raises
        return self.response


class TestPulseGenerator:
    @pytest.mark.asyncio
    async def test_basic_generation_uses_e4b(self) -> None:
        llm = _ScriptedLLM(
            response="*geht zur Kaffeemaschine und füllt sie auf*"
        )
        gen = PulseGenerator(llm_chat=llm, brain="e4b")
        mara = CharacterCard(id="m", name="Mara", persona="Pragmatischer Captain")
        drell = CharacterCard(id="d", name="Drell", persona="Schweigsamer Söldner")
        result = await gen.generate(
            character=mara,
            others_in_session=[mara, drell],
            recent_history=[
                {"role": "user", "name": "Mike", "content": "Was tust du?"},
                {"role": "character", "name": "Drell", "content": "*nickt nur*"},
            ],
        )
        assert "Kaffeemaschine" in result
        assert len(llm.calls) == 1
        assert llm.calls[0]["brain"] == "e4b"
        # The prompt must mention the other character so the LLM can address them.
        user_msg = next(
            m["content"] for m in llm.calls[0]["messages"] if m["role"] == "user"
        )
        assert "Drell" in user_msg

    @pytest.mark.asyncio
    async def test_state_is_threaded_into_prompt(self) -> None:
        llm = _ScriptedLLM(response="*seufzt*")
        gen = PulseGenerator(llm_chat=llm)
        mara = CharacterCard(
            id="m", name="Mara",
            state={"location": "Brücke", "mood": "müde", "last_action": "Kaffee"},
        )
        await gen.generate(
            character=mara, others_in_session=[mara], recent_history=[]
        )
        user_msg = next(
            m["content"] for m in llm.calls[0]["messages"] if m["role"] == "user"
        )
        assert "Brücke" in user_msg
        assert "müde" in user_msg

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_string(self) -> None:
        # The plugin uses the empty return as the trigger to fall back to
        # the static age-stage default — so this contract matters.
        llm = _ScriptedLLM(raises=RuntimeError("LLM exploded"))
        gen = PulseGenerator(llm_chat=llm)
        mara = CharacterCard(id="m", name="Mara")
        result = await gen.generate(
            character=mara, others_in_session=[mara], recent_history=[]
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_post_processing_strips_outer_quotes_and_state_block(self) -> None:
        llm = _ScriptedLLM(
            response='"*geht ins Bad* Bin gleich zurück." <state>location=Bad</state>'
        )
        gen = PulseGenerator(llm_chat=llm)
        mara = CharacterCard(id="m", name="Mara")
        result = await gen.generate(
            character=mara, others_in_session=[mara], recent_history=[]
        )
        assert "<state>" not in result
        # Pulse text doesn't carry state updates — they belong on real turns.
        assert "geht ins Bad" in result
        assert not result.startswith('"')

    @pytest.mark.asyncio
    async def test_post_processing_drops_pass_marker(self) -> None:
        llm = _ScriptedLLM(response="[PASS]")
        gen = PulseGenerator(llm_chat=llm)
        mara = CharacterCard(id="m", name="Mara")
        result = await gen.generate(
            character=mara, others_in_session=[mara], recent_history=[]
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_strips_leading_speaker_label(self) -> None:
        llm = _ScriptedLLM(response="Mara: *steht auf und reckt sich*")
        gen = PulseGenerator(llm_chat=llm)
        mara = CharacterCard(id="m", name="Mara")
        result = await gen.generate(
            character=mara, others_in_session=[mara], recent_history=[]
        )
        assert result.startswith("*")
        assert "Mara:" not in result


# ─── Phase D — Pulse-mention propagation via the orchestrator ────────────


class TestPulseMentionPropagation:
    """The plugin populates ``GroupTurnRequest.extra_forced``; the
    orchestrator must respect that order so the addressed character
    speaks in the same round as the pulse."""

    @pytest.mark.asyncio
    async def test_extra_forced_speakers_run_after_pulse_in_order(self) -> None:
        # We don't want to involve the LLM-order picker; round-robin keeps
        # this test deterministic without a real ``_ask_llm_for_order`` call.
        async def fake_llm(**_kwargs: Any) -> str:
            # Each character produces one short turn; routing isn't important.
            return "*reagiert kurz*"

        mara = CharacterCard(id="m", name="Mara")
        drell = CharacterCard(id="d", name="Drell")
        iko = CharacterCard(id="i", name="Iko")

        orch = GroupTurnOrchestrator(
            llm_chat=fake_llm,
            brain="e4b",
            max_tokens=80,
            turn_selection="round_robin",
            max_speakers_per_round=4,
        )
        req = GroupTurnRequest(
            session_id="s",
            history=[],
            characters=[mara, drell, iko],
            user_message="",
            pulse_from_id="m",
            pulse_text="Drell, hast du den Antrieb gecheckt?",
            # The plugin's pulse-mention propagation populates this:
            extra_forced=["d"],
        )
        result = await orch.run_round(req)
        # Drell goes FIRST — he was forced. Iko follows via round-robin.
        # Mara is excluded because she's the pulse_from_id (already "spoke").
        assert result.speaker_order[0] == "d"
        assert "m" not in result.speaker_order

    @pytest.mark.asyncio
    async def test_pulse_from_id_filtered_from_extra_forced(self) -> None:
        # Defence in depth: even if the propagation parser somehow returns
        # the pulser themselves, the orchestrator must drop the duplicate.
        async def fake_llm(**_kwargs: Any) -> str:
            return "*ok*"

        mara = CharacterCard(id="m", name="Mara")
        drell = CharacterCard(id="d", name="Drell")
        orch = GroupTurnOrchestrator(
            llm_chat=fake_llm,
            brain="e4b",
            max_tokens=80,
            turn_selection="round_robin",
        )
        req = GroupTurnRequest(
            session_id="s",
            history=[],
            characters=[mara, drell],
            pulse_from_id="m",
            pulse_text="...",
            extra_forced=["m", "d"],  # 'm' must be filtered out
        )
        result = await orch.run_round(req)
        assert "m" not in result.speaker_order
        assert "d" in result.speaker_order
