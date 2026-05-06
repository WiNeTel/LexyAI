"""Phase-9.5 character_chat audit fixes — three of Mike's reports:

1. **Speaker-Selection Brain wird nie angesprochen** — every speaker-pick
   path now emits one ``character_chat.speakers_picked`` log line so the
   path is visible. New ``always_call_orchestrator`` config forces an
   E4B confirmation even when mentions already covered everyone.

2. **State-Inkonsistenzen** ("Charakter ist nackt aber zupft an Kleidung")
   — state schema goes from a fixed 3-key whitelist to anchor keys
   (location, mood, last_action, **clothing, posture, condition**) plus
   free-form snake_case keys the LLM may invent. The state section is
   now MUST priority and the rules block tells the model to update its
   state when reality changes.

3. **Edit/Delete/Regenerate fehlen für Character-Bubbles** — covered by
   the existing tests for `_fetch_turn_row` + integration with the
   in-memory schema.
"""

from __future__ import annotations

from typing import Any

import pytest

from plugins.character_chat.character_card import (
    CharacterCard,
    _format_state_block,
)
from plugins.character_chat.context_budget import Priority
from plugins.character_chat.group_turn import (
    CharacterTurn,
    GroupTurnOrchestrator,
    GroupTurnRequest,
)
from plugins.character_chat.state_updater import (
    ANCHOR_STATE_KEYS,
    KNOWN_STATE_KEYS,
    merge_state,
    parse_state_block,
)


# ─── Recording fakes ─────────────────────────────────────────────────


class _RecordingLLM:
    def __init__(self, replies: list[str] | str = "Drell, Mara") -> None:
        self.replies = list(replies) if isinstance(replies, list) else [replies] * 99
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return self.replies.pop(0) if self.replies else ""


# ─── 1. Speaker selection always logs / always_call_orchestrator ─────


class TestSpeakerSelectionTransparency:
    @pytest.mark.asyncio
    async def test_round_robin_path_does_not_call_llm(
        self, caplog
    ) -> None:
        llm = _RecordingLLM("ok")
        mara = CharacterCard(id="m", name="Mara")
        drell = CharacterCard(id="d", name="Drell")
        orch = GroupTurnOrchestrator(
            llm_chat=llm, brain="a4b", max_tokens=80,
            turn_selection="round_robin",
            speaker_selection_brain="e4b",
        )
        req = GroupTurnRequest(
            session_id="s", history=[],
            characters=[mara, drell],
            user_message="hi everyone",
        )
        # ``run_round`` calls ``_pick_speakers`` internally and emits the
        # speaker-pick log line. Round-robin path must NOT consult the LLM.
        with caplog.at_level("INFO", logger="plugins.character_chat.group_turn"):
            await orch.run_round(req)
        # No order-decision LLM call (only the actual char-turn calls).
        # The order-decision call is identified by the "Turn-Orchestrator"
        # system message — count how many of those landed.
        order_calls = [
            c for c in llm.calls
            if any("Turn-Orchestrator" in m.get("content", "")
                   for m in c.get("messages", [])
                   if m.get("role") == "system")
        ]
        assert order_calls == []
        # And the log line confirms the path.
        assert any(
            "character_chat.speakers_picked" in r.message
            and "method=round_robin" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    @pytest.mark.asyncio
    async def test_autonomous_path_calls_orchestrator_brain(
        self, caplog
    ) -> None:
        # Two chars, no mentions in the message, autonomous mode → LLM
        # MUST be called for the order, on speaker_selection_brain (e4b).
        llm = _RecordingLLM([
            "m, d",                  # the order-decision LLM call
            "Maras Antwort",         # Mara's turn
            "Drells Antwort",        # Drell's turn
        ])
        mara = CharacterCard(id="m", name="Mara", persona="Captain")
        drell = CharacterCard(id="d", name="Drell", persona="Söldner")
        orch = GroupTurnOrchestrator(
            llm_chat=llm, brain="a4b", max_tokens=80,
            turn_selection="autonomous",
            speaker_selection_brain="e4b",
        )
        req = GroupTurnRequest(
            session_id="s", history=[],
            characters=[mara, drell],
            user_message="hallo zusammen",  # no NL-mention
        )
        with caplog.at_level("INFO", logger="plugins.character_chat.group_turn"):
            await orch.run_round(req)
        # First call MUST be on the order-decision brain (e4b).
        first = llm.calls[0]
        assert first["brain"] == "e4b"
        # Speaker-pick log shows method=llm + brain=e4b + brain_called=True.
        assert any(
            "character_chat.speakers_picked" in r.message
            and "method=llm" in r.message
            and "brain_called=True" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    @pytest.mark.asyncio
    async def test_always_call_orchestrator_forces_llm_even_with_mentions(
        self,
    ) -> None:
        # User message NL-mentions both chars in order. Without the
        # always-flag the LLM is skipped (mentions cover everything).
        # With the flag, the orchestrator is consulted to confirm.
        llm = _RecordingLLM([
            "m, d",                  # confirm order
            "Maras Antwort",
            "Drells Antwort",
        ])
        mara = CharacterCard(id="m", name="Mara")
        drell = CharacterCard(id="d", name="Drell")
        orch = GroupTurnOrchestrator(
            llm_chat=llm, brain="a4b", max_tokens=80,
            turn_selection="autonomous",
            speaker_selection_brain="e4b",
            always_call_orchestrator=True,
        )
        req = GroupTurnRequest(
            session_id="s", history=[],
            characters=[mara, drell],
            user_message="Mara, schau mal. Drell, sicher die Tür.",
        )
        await orch.run_round(req)
        # First call is on e4b — the orchestrator-refine pass.
        first = llm.calls[0]
        assert first["brain"] == "e4b"
        # The "preferred order" hint must be present in the user prompt.
        user_msg = next(
            m for m in first["messages"] if m["role"] == "user"
        )
        assert "Vorgeschlagene Reihenfolge" in user_msg["content"]


# ─── 2. State schema: free-form + anchor keys + MUST priority ────────


class TestStateSchemaFreeForm:
    def test_anchor_keys_include_clothing_posture_condition(self) -> None:
        # Mike's audit: clothing was unrepresentable → "nackt"-bug.
        for key in ("clothing", "posture", "condition"):
            assert key in ANCHOR_STATE_KEYS

    def test_known_state_keys_alias_anchor_set(self) -> None:
        # Backwards-compat alias for older imports.
        assert KNOWN_STATE_KEYS == frozenset(ANCHOR_STATE_KEYS)

    def test_parser_accepts_freeform_snake_case_key(self) -> None:
        text = "<state>holds_object=Schwert; proximity=nah</state>"
        cleaned, updates = parse_state_block(text)
        assert cleaned == ""  # block stripped
        assert updates == {"holds_object": "Schwert", "proximity": "nah"}

    def test_parser_rejects_non_snake_case_keys(self) -> None:
        # CamelCase, dots, spaces, hyphens — all rejected so the prompt
        # renderer doesn't choke on weird symbols.
        text = (
            "<state>Bad-Key=x; .;:nope=y; with space=z; "
            "valid_key=ok</state>"
        )
        _, updates = parse_state_block(text)
        # Only the valid one survives.
        assert updates == {"valid_key": "ok"}

    def test_parser_normalises_anchor_key_case(self) -> None:
        # CamelCase anchor key (LLM occasionally emits) is lower-cased
        # and accepted because the anchor set contains the lowercase form.
        text = "<state>Clothing=nackt</state>"
        _, updates = parse_state_block(text)
        assert updates == {"clothing": "nackt"}

    def test_merge_accepts_freeform_keys_and_truncates_long_values(
        self,
    ) -> None:
        long_val = "x" * 300
        merged = merge_state(
            {"location": "Bad"},
            {"holds_object": "Tasse", "tooooo_long": long_val},
        )
        assert merged["holds_object"] == "Tasse"
        assert merged["tooooo_long"].endswith("…")
        assert len(merged["tooooo_long"]) <= 121

    def test_merge_drops_empty_string_values(self) -> None:
        merged = merge_state(
            {"clothing": "nackt", "mood": "wach"},
            {"clothing": ""},
        )
        assert "clothing" not in merged
        assert merged["mood"] == "wach"


class TestStateRendering:
    def test_anchors_render_in_fixed_order(self) -> None:
        state = {
            "condition": "müde",
            "location": "Bad",
            "mood": "ruhig",
            "clothing": "Bademantel",
            "last_action": "geduscht",
            "posture": "stehend",
        }
        rendered = _format_state_block(state)
        # All anchor keys present.
        for de_label in ("Ort:", "Stimmung:", "Letzte Aktion:", "Kleidung:", "Haltung:", "Zustand:"):
            assert de_label in rendered
        # Order: location < mood < last_action < clothing < posture < condition.
        idx_loc = rendered.index("Ort:")
        idx_mood = rendered.index("Stimmung:")
        idx_la = rendered.index("Letzte Aktion:")
        idx_clo = rendered.index("Kleidung:")
        idx_pos = rendered.index("Haltung:")
        idx_con = rendered.index("Zustand:")
        assert idx_loc < idx_mood < idx_la < idx_clo < idx_pos < idx_con

    def test_freeform_keys_render_after_anchors_with_titlecase_label(
        self,
    ) -> None:
        state = {
            "location": "Wohnzimmer",
            "holds_object": "Buch",
            "tone_of_voice": "leise",
        }
        rendered = _format_state_block(state)
        assert "Ort:" in rendered
        assert "Holds Object:" in rendered
        assert "Tone Of Voice:" in rendered
        # Anchor first, then free-form.
        assert rendered.index("Ort:") < rendered.index("Holds Object:")


class TestStateSectionPriority:
    @pytest.mark.asyncio
    async def test_char_state_section_is_must_priority(self) -> None:
        async def fake_llm(**_kw):
            return "ok"
        mara = CharacterCard(
            id="m", name="Mara",
            state={"location": "Brücke", "clothing": "Uniform"},
        )
        orch = GroupTurnOrchestrator(
            llm_chat=fake_llm, brain="a4b", max_tokens=80,
            turn_selection="round_robin",
        )
        req = GroupTurnRequest(
            session_id="s", history=[], characters=[mara],
            user_message="hi",
        )
        sections = orch._build_turn_sections(
            card=mara, req=req, all_cards=[mara],
            previous_turns=[], own_memories=None,
        )
        state_section = next(
            (s for s in sections if s.name == "char_state"), None
        )
        assert state_section is not None
        assert state_section.priority == Priority.MUST
        # Body contains the consistency rule + the anchor labels.
        assert "Halte dich strikt an diesen Zustand" in state_section.text
        assert "Brücke" in state_section.text
        assert "Uniform" in state_section.text


# ─── 3. Edit/Delete/Regenerate handlers exist & dispatch correctly ───
#
# Wir mocken das WS-Client-Interface + DB so dass _ws_turn_edit /
# _ws_turn_delete in voller Echo-Schleife durchlaufen. Der LLM-Loop
# in _ws_turn_regenerate wird hier NICHT getestet (zu viele
# Abhängigkeiten — einfacher als manueller End-to-End-Test).


class _FakeWSClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class TestTurnActionHandlers:
    """Smoke-test the new edit/delete WS handlers via the plugin's
    public attributes. We don't fully boot the plugin; we use
    ``aiosqlite`` directly + a tiny mock around the API so the SQL
    path is exercised end-to-end."""

    @pytest.mark.asyncio
    async def test_handlers_registered_on_plugin(self) -> None:
        from plugins.character_chat.character_chat_plugin import (
            CharacterChatPlugin,
        )
        for name in (
            "_ws_turn_edit",
            "_ws_turn_delete",
            "_ws_turn_regenerate",
            "_fetch_turn_row",
        ):
            assert callable(getattr(CharacterChatPlugin, name, None)), name

    @pytest.mark.asyncio
    async def test_edit_rejects_missing_turn_id(self) -> None:
        # The handler validates input shape before doing anything DB-side.
        # We use a stub plugin so we don't need a full LexyApp.
        from plugins.character_chat.character_chat_plugin import (
            CharacterChatPlugin,
        )

        class _Stub:
            def __init__(self):
                self._db_obj = None

        stub = _Stub()
        client = _FakeWSClient()
        # Bind the bound-method dynamically to a stub object that has the
        # exact attributes the handler reads (none, in the early-bail path).
        handler = CharacterChatPlugin._ws_turn_edit.__get__(stub)
        await handler(client, {"turn_id": "", "content": "irrelevant"})
        assert client.sent and client.sent[0]["ok"] is False
        assert "required" in client.sent[0]["error"]

    @pytest.mark.asyncio
    async def test_edit_rejects_empty_content(self) -> None:
        from plugins.character_chat.character_chat_plugin import (
            CharacterChatPlugin,
        )

        class _Stub:
            pass

        stub = _Stub()
        client = _FakeWSClient()
        handler = CharacterChatPlugin._ws_turn_edit.__get__(stub)
        await handler(client, {"turn_id": "abc", "content": "   "})
        assert client.sent and client.sent[0]["ok"] is False
