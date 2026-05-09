"""
Phase 13.5 hotfix — pin two things:

1. ``_looks_imperative_template`` correctly classifies the Castaway
   scenario's action-discipline prompts as instructions (not chat
   text). Those prompts (e.g. "Lena REAGIERT auf einen anderen
   Charakter NAMENTLICH. Wähle EINS: ...") were being persisted as
   visible chat turns by the original 13.5 (A) fix — a regression
   we're closing here.

2. ``PulseGenerator.generate`` accepts a ``style_guidance`` kwarg and
   passes it into the user prompt under "Aktions-Vorgabe". The
   generator's job is to produce NARRATIVE output that follows the
   guidance — not echo the guidance itself.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from plugins.character_chat.character_card import CharacterCard
from plugins.character_chat.character_chat_plugin import (
    _looks_imperative_template,
)
from plugins.character_chat.pulse_generator import PulseGenerator


def _card(name: str = "Lena", pulse_prompt: str = "") -> CharacterCard:
    return CharacterCard(
        name=name,
        persona=f"{name} ist {name}.",
        proactive_pulse_prompt=pulse_prompt,
        age_stage="adult",
        created_at=time.time(),
        updated_at=time.time(),
    )


# ─── Imperative-template detection ──────────────────────────────────


class TestImperativeDetection:
    """The heuristic that distinguishes LLM instructions from narrative
    pulse text. False positives here turn an author's narrative pulse
    into a generator-driven one (small loss). False negatives leak the
    template verbatim into chat (big loss). We tune for the latter."""

    def test_castaway_lena_template_detected(self) -> None:
        # Verbatim from seed_castaway_scenario.py.
        prompt = (
            "Lena REAGIERT auf einen anderen Charakter NAMENTLICH. "
            "Wähle EINS: spricht Sandra direkt an, folgt Mira zum "
            "Wald, fragt Yara was zu tun ist. Verbote: KEIN passives "
            "Sand-Anstarren."
        )
        assert _looks_imperative_template(prompt) is True

    def test_castaway_sandra_template_detected(self) -> None:
        prompt = (
            "Sandra MACHT eine konkrete Aktion statt nur zu fühlen. "
            "Wähle EINS: prüft die anderen auf Verletzungen, geht "
            "zum Süßwasser-Bach. ABSOLUTES VERBOT von 'starre auf "
            "Sand'."
        )
        assert _looks_imperative_template(prompt) is True

    def test_castaway_mira_template_detected(self) -> None:
        prompt = (
            "Mira ist SCHON UNTERWEGS oder kommt gerade zurück. "
            "Wähle EINS und beschreibe es konkret: kommt mit einer "
            "Banane / Mango / Kokosnuss aus dem Wald zurück."
        )
        assert _looks_imperative_template(prompt) is True

    def test_narrative_pulse_passes_through(self) -> None:
        """Hand-written narrative pulses (the ORIGINAL design intent
        of proactive_pulse_prompt) must NOT be flagged."""
        narrative = (
            "*hört Schritte am Strand und blickt vorsichtig auf, "
            "die Hände noch in den Sand gegraben*"
        )
        assert _looks_imperative_template(narrative) is False

    def test_empty_returns_false(self) -> None:
        assert _looks_imperative_template("") is False

    def test_dialog_only_passes(self) -> None:
        assert _looks_imperative_template(
            '"Sandra? Bist du da?" *flüstert leise*'
        ) is False


# ─── PulseGenerator.style_guidance ──────────────────────────────────


class _FakeLLM:
    def __init__(self, reply: str = "*hebt eine Muschel auf* 'Schau mal.'"):
        self.reply = reply
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


class TestStyleGuidance:
    def test_style_guidance_lands_in_user_prompt(self) -> None:
        llm = _FakeLLM()
        gen = PulseGenerator(llm_chat=llm, brain="e4b")
        card = _card("Mira")
        guidance = (
            "Mira ist SCHON UNTERWEGS. Wähle EINS: kommt mit Kokosnuss "
            "zurück. Verbote: KEIN Sand-Anstarren."
        )
        text = asyncio.run(gen.generate(
            character=card,
            others_in_session=[],
            recent_history=[],
            style_guidance=guidance,
        ))
        # Generator returned the LLM's narrative output (NOT the
        # guidance verbatim).
        assert text.startswith("*hebt eine Muschel auf*") or "Schau mal" in text
        # The guidance was passed into the user prompt under the
        # 'Aktions-Vorgabe' section.
        user_prompt = llm.calls[0]["messages"][1]["content"]
        assert "Aktions-Vorgabe" in user_prompt
        assert "Wähle EINS: kommt mit Kokosnuss" in user_prompt
        # And the prompt explicitly tells the LLM to write narrative,
        # not echo the instruction.
        assert "NARRATIV" in user_prompt or "narrativ" in user_prompt.lower()

    def test_no_style_guidance_omits_section(self) -> None:
        llm = _FakeLLM()
        gen = PulseGenerator(llm_chat=llm, brain="e4b")
        card = _card("Mira")
        asyncio.run(gen.generate(
            character=card,
            others_in_session=[],
            recent_history=[],
            # no style_guidance
        ))
        user_prompt = llm.calls[0]["messages"][1]["content"]
        assert "Aktions-Vorgabe" not in user_prompt

    def test_empty_style_guidance_treated_as_none(self) -> None:
        llm = _FakeLLM()
        gen = PulseGenerator(llm_chat=llm, brain="e4b")
        card = _card("Mira")
        asyncio.run(gen.generate(
            character=card,
            others_in_session=[],
            recent_history=[],
            style_guidance="   ",  # whitespace only
        ))
        user_prompt = llm.calls[0]["messages"][1]["content"]
        assert "Aktions-Vorgabe" not in user_prompt
