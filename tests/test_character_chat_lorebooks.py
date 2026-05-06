"""Tests for the Lorebook subsystem (Phase 9.8).

Mike asked for SillyTavern-style Lorebooks: per-character / per-session
/ global collections of triggered lore entries that get injected into
the prompt when the right keywords show up in chat.

Three layers under test:

1. **Store** (``LorebookStore``) — CRUD + scope semantics + cascade-on-
   delete + entry-without-keys-or-always_on rejected.
2. **Engine** (``LorebookEngine``) — substring trigger scan with
   word-boundary on single-word keys, always_on always fires, scope
   filter, per-book token budget, position-based grouping.
3. **Group-turn integration** — when ``req.lore_by_speaker[char_id]``
   is set, the prompt builder injects sections at the right slots
   (``lorebook_before_persona`` etc.) with HIGH priority.
"""

from __future__ import annotations

from typing import Any

import aiosqlite
import pytest
import pytest_asyncio

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.context_budget import Priority
from plugins.character_chat.group_turn import (
    GroupTurnOrchestrator,
    GroupTurnRequest,
)
from plugins.character_chat.lorebook_engine import (
    ActivationResult,
    LorebookEngine,
    SECTION_NAME_PER_POSITION,
    render_position_block,
)
from plugins.character_chat.lorebook_store import (
    DEFAULT_TOKEN_BUDGET,
    LorebookStore,
    POSITION_AFTER_PERSONA,
    POSITION_BEFORE_HISTORY,
    POSITION_BEFORE_PERSONA,
    POSITION_BEFORE_SCENARIO,
    POSITION_BEFORE_USER_MESSAGE,
    SCOPE_CHARACTER,
    SCOPE_GLOBAL,
    SCOPE_SESSION,
)


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store() -> LorebookStore:
    db = await aiosqlite.connect(":memory:")
    s = LorebookStore(db)
    await s.init_schema()
    yield s
    await db.close()


# ─── 1. Store ────────────────────────────────────────────────────────


class TestLorebookStore:
    @pytest.mark.asyncio
    async def test_create_global_lorebook(self, store: LorebookStore) -> None:
        book = await store.create_lorebook(
            name="Welt", description="Hauptwelt-Lore",
        )
        assert book.scope == SCOPE_GLOBAL
        assert book.scope_id == ""
        assert book.token_budget == DEFAULT_TOKEN_BUDGET
        assert book.enabled

    @pytest.mark.asyncio
    async def test_create_character_lorebook_requires_scope_id(
        self, store: LorebookStore
    ) -> None:
        with pytest.raises(ValueError, match="requires a scope_id"):
            await store.create_lorebook(
                name="Mara-Lore", scope=SCOPE_CHARACTER,
            )

    @pytest.mark.asyncio
    async def test_create_session_lorebook_requires_scope_id(
        self, store: LorebookStore
    ) -> None:
        with pytest.raises(ValueError, match="requires a scope_id"):
            await store.create_lorebook(name="Sess", scope=SCOPE_SESSION)

    @pytest.mark.asyncio
    async def test_unknown_scope_rejected(self, store: LorebookStore) -> None:
        with pytest.raises(ValueError, match="unknown scope"):
            await store.create_lorebook(name="x", scope="cosmic")

    @pytest.mark.asyncio
    async def test_global_scope_clears_scope_id(self, store: LorebookStore) -> None:
        # Caller passing scope_id with scope=global is harmless — we drop it.
        book = await store.create_lorebook(
            name="W", scope=SCOPE_GLOBAL, scope_id="ignored",
        )
        assert book.scope_id == ""

    @pytest.mark.asyncio
    async def test_list_filters_by_scope_and_scope_id(
        self, store: LorebookStore
    ) -> None:
        b1 = await store.create_lorebook(name="A")  # global
        b2 = await store.create_lorebook(
            name="B", scope=SCOPE_CHARACTER, scope_id="char-1"
        )
        b3 = await store.create_lorebook(
            name="C", scope=SCOPE_CHARACTER, scope_id="char-2"
        )
        b4 = await store.create_lorebook(
            name="D", scope=SCOPE_SESSION, scope_id="sess-x",
        )
        # Filter by scope
        chars_only = await store.list_lorebooks(scope=SCOPE_CHARACTER)
        assert {b.id for b in chars_only} == {b2.id, b3.id}
        # Filter by scope + scope_id
        sess = await store.list_lorebooks(
            scope=SCOPE_SESSION, scope_id="sess-x"
        )
        assert {b.id for b in sess} == {b4.id}
        # All
        all_ = await store.list_lorebooks()
        assert len(all_) == 4

    @pytest.mark.asyncio
    async def test_update_lorebook(self, store: LorebookStore) -> None:
        book = await store.create_lorebook(name="Welt")
        updated = await store.update_lorebook(
            book.id, name="Welt v2", token_budget=500, enabled=False,
        )
        assert updated is not None
        assert updated.name == "Welt v2"
        assert updated.token_budget == 500
        assert updated.enabled is False

    @pytest.mark.asyncio
    async def test_delete_cascades_to_entries(self, store: LorebookStore) -> None:
        book = await store.create_lorebook(name="Welt")
        await store.create_entry(
            lorebook_id=book.id, name="E1", keys=["x"], content="x",
        )
        await store.create_entry(
            lorebook_id=book.id, name="E2", keys=["y"], content="y",
        )
        ok = await store.delete_lorebook(book.id)
        assert ok is True
        remaining = await store.list_entries(lorebook_id=book.id)
        assert remaining == []

    @pytest.mark.asyncio
    async def test_create_entry_requires_keys_or_always_on(
        self, store: LorebookStore
    ) -> None:
        book = await store.create_lorebook(name="Welt")
        with pytest.raises(ValueError, match="at least one key"):
            await store.create_entry(
                lorebook_id=book.id, name="E", content="...",
                keys=[],  # empty
                always_on=False,
            )

    @pytest.mark.asyncio
    async def test_create_entry_unknown_lorebook_rejected(
        self, store: LorebookStore
    ) -> None:
        with pytest.raises(ValueError, match="lorebook not found"):
            await store.create_entry(
                lorebook_id="ghost", name="E", keys=["x"], content="x",
            )

    @pytest.mark.asyncio
    async def test_unknown_position_rejected(
        self, store: LorebookStore
    ) -> None:
        book = await store.create_lorebook(name="Welt")
        with pytest.raises(ValueError, match="unknown position"):
            await store.create_entry(
                lorebook_id=book.id, name="E", keys=["x"],
                content="x", position="nowhere",
            )

    @pytest.mark.asyncio
    async def test_update_entry_keys(self, store: LorebookStore) -> None:
        book = await store.create_lorebook(name="W")
        entry = await store.create_entry(
            lorebook_id=book.id, name="E", keys=["alpha"], content="a",
        )
        updated = await store.update_entry(
            entry.id, keys=["alpha", "beta", "  "], content="b",
        )
        assert updated is not None
        # Empty/whitespace keys filtered out.
        assert updated.keys == ["alpha", "beta"]
        assert updated.content == "b"

    @pytest.mark.asyncio
    async def test_round_trip_keys_via_json(self, store: LorebookStore) -> None:
        book = await store.create_lorebook(name="W")
        entry = await store.create_entry(
            lorebook_id=book.id, name="E",
            keys=["mit Leerzeichen", "umlaut-ä", "x"],
            content="x",
        )
        loaded = await store.get_entry(entry.id)
        assert loaded is not None
        assert loaded.keys == ["mit Leerzeichen", "umlaut-ä", "x"]


# ─── 2. Engine ──────────────────────────────────────────────────────


class TestLorebookEngineActivation:
    def setup_method(self) -> None:
        self.engine = LorebookEngine()
        self.mara = CharacterCard(id="mara", name="Mara", persona="Captain")
        self.drell = CharacterCard(id="drell", name="Drell", persona="Söldner")

    @pytest.mark.asyncio
    async def test_substring_match_fires_entry(
        self, store: LorebookStore
    ) -> None:
        book = await store.create_lorebook(name="W")
        await store.create_entry(
            lorebook_id=book.id, name="Drache", keys=["Vyrkos"],
            content="Vyrkos ist alt.",
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        result = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="Wer ist Vyrkos eigentlich?",
            pulse_text="", lorebooks=books, entries=entries,
        )
        items = result.all_items()
        assert len(items) == 1
        assert items[0].name == "Drache"
        assert items[0].matched_key == "Vyrkos"

    @pytest.mark.asyncio
    async def test_word_boundary_for_single_word_keys(
        self, store: LorebookStore
    ) -> None:
        # "rage" must NOT fire on "courage".
        book = await store.create_lorebook(name="W")
        await store.create_entry(
            lorebook_id=book.id, name="Rage-Lore", keys=["rage"],
            content="x",
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        # "courage" should NOT trigger.
        no = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="It takes courage.", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert no.all_items() == []
        # Bare "rage" SHOULD trigger.
        yes = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="Pure rage in his eyes.", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert len(yes.all_items()) == 1

    @pytest.mark.asyncio
    async def test_phrase_keys_use_substring(
        self, store: LorebookStore
    ) -> None:
        # Multi-word keys fall back to substring (word-boundary on
        # phrases is fiddly and the convenience matters).
        book = await store.create_lorebook(name="W")
        await store.create_entry(
            lorebook_id=book.id, name="Ph", keys=["Black Stone"],
            content="x",
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        result = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="The Black Stone glows.", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert len(result.all_items()) == 1

    @pytest.mark.asyncio
    async def test_always_on_fires_without_match(
        self, store: LorebookStore
    ) -> None:
        book = await store.create_lorebook(name="W")
        await store.create_entry(
            lorebook_id=book.id, name="Welt", always_on=True,
            content="Mittelalter-Fantasy.", keys=[],
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        result = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="Hi.", pulse_text="",
            lorebooks=books, entries=entries,
        )
        items = result.all_items()
        assert len(items) == 1
        assert items[0].matched_key == ""

    @pytest.mark.asyncio
    async def test_disabled_book_skipped(
        self, store: LorebookStore
    ) -> None:
        b1 = await store.create_lorebook(name="enabled")
        b2 = await store.create_lorebook(name="disabled")
        await store.update_lorebook(b2.id, enabled=False)
        await store.create_entry(
            lorebook_id=b1.id, name="A", keys=["x"], content="a",
        )
        await store.create_entry(
            lorebook_id=b2.id, name="B", keys=["x"], content="b",
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        result = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="x", pulse_text="",
            lorebooks=books, entries=entries,
        )
        names = {it.name for it in result.all_items()}
        assert names == {"A"}  # only the enabled book's entry fires

    @pytest.mark.asyncio
    async def test_character_scope_filter(
        self, store: LorebookStore
    ) -> None:
        # Char-scoped book for Mara only — must not fire for Drell.
        b = await store.create_lorebook(
            name="MaraLore", scope=SCOPE_CHARACTER, scope_id="mara",
        )
        await store.create_entry(
            lorebook_id=b.id, name="Mara-Secret", always_on=True,
            content="...", keys=[],
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        # For Mara — fires.
        ok = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="hi", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert len(ok.all_items()) == 1
        # For Drell — drops the book entirely.
        skip = self.engine.activate(
            speaker=self.drell, session_id="s1", history=[],
            user_message="hi", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert skip.all_items() == []

    @pytest.mark.asyncio
    async def test_session_scope_filter(
        self, store: LorebookStore
    ) -> None:
        b = await store.create_lorebook(
            name="SessLore", scope=SCOPE_SESSION, scope_id="sess-A",
        )
        await store.create_entry(
            lorebook_id=b.id, name="ThisSess", always_on=True,
            content="...", keys=[],
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        for_a = self.engine.activate(
            speaker=self.mara, session_id="sess-A", history=[],
            user_message="x", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert len(for_a.all_items()) == 1
        for_b = self.engine.activate(
            speaker=self.mara, session_id="sess-B", history=[],
            user_message="x", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert for_b.all_items() == []

    @pytest.mark.asyncio
    async def test_priority_ordering(
        self, store: LorebookStore
    ) -> None:
        b = await store.create_lorebook(name="W")
        await store.create_entry(
            lorebook_id=b.id, name="Z-late", keys=["x"], content="z",
            priority=300,
        )
        await store.create_entry(
            lorebook_id=b.id, name="A-early", keys=["x"], content="a",
            priority=10,
        )
        await store.create_entry(
            lorebook_id=b.id, name="M-mid", keys=["x"], content="m",
            priority=100,
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        result = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="x", pulse_text="",
            lorebooks=books, entries=entries,
        )
        names = [it.name for it in result.all_items()]
        assert names == ["A-early", "M-mid", "Z-late"]

    @pytest.mark.asyncio
    async def test_token_budget_drops_overflow(
        self, store: LorebookStore
    ) -> None:
        # Tiny budget — only the first entry fits.
        b = await store.create_lorebook(name="W", token_budget=10)
        await store.create_entry(
            lorebook_id=b.id, name="First", always_on=True,
            content="A" * 30, keys=[], priority=10,
        )
        await store.create_entry(
            lorebook_id=b.id, name="Second", always_on=True,
            content="B" * 30, keys=[], priority=20,
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        result = self.engine.activate(
            speaker=self.mara, session_id="s1", history=[],
            user_message="x", pulse_text="",
            lorebooks=books, entries=entries,
        )
        # First entry always lands even when over budget; second is
        # dropped because the budget is already consumed.
        names = [it.name for it in result.all_items()]
        assert names == ["First"]
        assert result.skipped_budget == 1

    @pytest.mark.asyncio
    async def test_scan_depth_limits_history(
        self, store: LorebookStore
    ) -> None:
        b = await store.create_lorebook(name="W")
        await store.create_entry(
            lorebook_id=b.id, name="E", keys=["secret"], content="x",
            scan_depth=2,  # only last 2 messages
        )
        books = await store.list_lorebooks(enabled_only=True)
        entries = {b.id: await store.list_entries(lorebook_id=b.id) for b in books}
        # "secret" appears only in the OLDEST message → out of scan window.
        history = [
            {"role": "user", "content": "the secret was buried"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "later that day"},
            {"role": "assistant", "content": "alright"},
        ]
        miss = self.engine.activate(
            speaker=self.mara, session_id="s1", history=history,
            user_message="okay", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert miss.all_items() == []
        # Now the user message contains "secret" — within scan window.
        hit = self.engine.activate(
            speaker=self.mara, session_id="s1", history=history,
            user_message="tell me the secret", pulse_text="",
            lorebooks=books, entries=entries,
        )
        assert len(hit.all_items()) == 1


# ─── 3. Group-turn integration ──────────────────────────────────────


class TestGroupTurnLoreInjection:
    @pytest.mark.asyncio
    async def test_injects_lore_section_in_correct_position(self) -> None:
        # Build an ActivationResult by hand (we test the engine separately).
        from plugins.character_chat.lorebook_engine import ActivatedLore

        activation = ActivationResult()
        activation.by_position[POSITION_BEFORE_SCENARIO] = [
            ActivatedLore(
                entry_id="e1", name="Welt",
                content="Mittelalter-Fantasy.",
                position=POSITION_BEFORE_SCENARIO, priority=10,
            )
        ]

        async def fake_llm(**_kw: Any) -> str:
            return "ok"

        mara = CharacterCard(id="m", name="Mara", persona="Captain")
        orch = GroupTurnOrchestrator(
            llm_chat=fake_llm, brain="a4b", max_tokens=80,
            turn_selection="round_robin",
        )
        req = GroupTurnRequest(
            session_id="s", history=[], characters=[mara],
            user_message="hi",
            lore_by_speaker={"m": activation},
        )
        sections = orch._build_turn_sections(
            card=mara, req=req, all_cards=[mara],
            previous_turns=[], own_memories=None,
        )
        names = [s.name for s in sections]
        assert "lorebook_before_scenario" in names
        # Lore section must come BEFORE scenario / age_guidance / others.
        idx_lore = names.index("lorebook_before_scenario")
        # Persona is in the system stack; lore should sit between
        # persona and scenario (or earlier, depending on whether the
        # character has a non-empty persona/scenario).
        # We assert lore precedes "rules" — that's the strongest
        # invariant that always holds.
        assert idx_lore < names.index("rules")

    @pytest.mark.asyncio
    async def test_no_lore_for_speaker_no_section(self) -> None:
        async def fake_llm(**_kw: Any) -> str:
            return "ok"
        mara = CharacterCard(id="m", name="Mara", persona="Captain")
        orch = GroupTurnOrchestrator(
            llm_chat=fake_llm, brain="a4b", max_tokens=80,
            turn_selection="round_robin",
        )
        req = GroupTurnRequest(
            session_id="s", history=[], characters=[mara],
            user_message="hi",
            lore_by_speaker={},  # empty
        )
        sections = orch._build_turn_sections(
            card=mara, req=req, all_cards=[mara],
            previous_turns=[], own_memories=None,
        )
        names = [s.name for s in sections]
        for sec in (
            "lorebook_before_persona",
            "lorebook_after_persona",
            "lorebook_before_scenario",
            "lorebook_before_history",
            "lorebook_before_user_message",
        ):
            assert sec not in names

    @pytest.mark.asyncio
    async def test_lore_section_priority_high(self) -> None:
        from plugins.character_chat.lorebook_engine import ActivatedLore

        activation = ActivationResult()
        activation.by_position[POSITION_BEFORE_SCENARIO] = [
            ActivatedLore(
                entry_id="e1", name="X", content="x",
                position=POSITION_BEFORE_SCENARIO, priority=10,
            )
        ]
        async def fake_llm(**_kw: Any) -> str:
            return "ok"
        mara = CharacterCard(id="m", name="Mara")
        orch = GroupTurnOrchestrator(
            llm_chat=fake_llm, brain="a4b", max_tokens=80,
            turn_selection="round_robin",
        )
        req = GroupTurnRequest(
            session_id="s", history=[], characters=[mara],
            user_message="hi",
            lore_by_speaker={"m": activation},
        )
        sections = orch._build_turn_sections(
            card=mara, req=req, all_cards=[mara],
            previous_turns=[], own_memories=None,
        )
        lore_sec = next(
            (s for s in sections if s.name == "lorebook_before_scenario"), None,
        )
        assert lore_sec is not None
        # HIGH so it survives most trimming, but not MUST so under hard
        # budget pressure it can still be reduced.
        assert lore_sec.priority == Priority.HIGH

    def test_render_position_block_format(self) -> None:
        from plugins.character_chat.lorebook_engine import ActivatedLore

        items = [
            ActivatedLore(
                entry_id="e1", name="Drache",
                content="Vyrkos ist alt.",
                position=POSITION_BEFORE_SCENARIO, priority=10,
            ),
            ActivatedLore(
                entry_id="e2", name="Ort",
                content="Schwarzwald.",
                position=POSITION_BEFORE_SCENARIO, priority=20,
            ),
        ]
        rendered = render_position_block(items)
        assert "## Lorebook" in rendered
        assert "### Drache" in rendered
        assert "Vyrkos ist alt." in rendered
        assert "### Ort" in rendered

    def test_section_name_per_position_complete(self) -> None:
        # Catches drift between the engine's positions and the orchestrator's
        # known section slots.
        for pos in (
            POSITION_BEFORE_PERSONA, POSITION_AFTER_PERSONA,
            POSITION_BEFORE_SCENARIO, POSITION_BEFORE_HISTORY,
            POSITION_BEFORE_USER_MESSAGE,
        ):
            assert pos in SECTION_NAME_PER_POSITION
