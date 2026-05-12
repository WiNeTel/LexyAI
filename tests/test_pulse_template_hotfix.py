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

    def test_case_insensitive_match(self) -> None:
        """Mike's leak: e4b echoed the guidance with mixed casing
        ('macht etwas BEOBACHTBARES' instead of the original
        'MACHT etwas Beobachtbares'). The detector must not be
        fooled by simple case shuffling."""
        # Lowercase 'macht', uppercase 'BEOBACHTBARES' — the exact
        # leaked pattern from Mike's chat.
        leaked = (
            "Yara macht etwas BEOBACHTBARES. Wähle EINS: zeigt mit "
            "dem Finger auf etwas Konkretes."
        )
        assert _looks_imperative_template(leaked) is True

    def test_kein_passives_marker(self) -> None:
        """Echo'd 'Verbote: KEIN passives ...' template fragment
        triggers via the strong 'verbote:' marker (and as a bonus
        also matches the 'kein passives' medium marker)."""
        partial = (
            "*sucht im Sand* 'Sandra...?' Verbote: KEIN passives "
            "Sand-Anstarren."
        )
        assert _looks_imperative_template(partial) is True

    def test_kein_schweigen_alone_does_not_trigger(self) -> None:
        """Phase 13.5 hotfix v5: 'kein schweigen' alone could
        plausibly appear in narrative ('Genug, kein Schweigen mehr.').
        The medium marker needs reinforcement (2+ hits) before the
        guard kicks in. This test ensures false positives don't
        block valid narrative."""
        narrative = (
            "*tritt einen Schritt vor und hebt den Kopf* "
            "'Genug, kein Schweigen mehr — ich rede jetzt.'"
        )
        # No strong marker, only 1 medium marker → not flagged.
        assert _looks_imperative_template(narrative) is False

    def test_two_medium_markers_trigger(self) -> None:
        """If both 'wähle eins' AND 'kein passives' appear in the
        SAME output, that's strong evidence of an echoed template
        and the guard fires."""
        partial = (
            "Du sollst etwas machen. Wähle eins von den Optionen. "
            "Kein passives Verhalten."
        )
        # 2 medium markers → triggers.
        assert _looks_imperative_template(partial) is True

    def test_narrative_with_ist_schon_unterwegs_not_flagged(self) -> None:
        """Phase 13.5 hotfix v5 — Mike's session log: e4b produced
        narrative containing 'ist schon unterwegs' ('Mira ist schon
        unterwegs zum Wald'). The previous heuristic caught it as
        imperative, replaced with default; the new heuristic lets
        it through because that phrase is plausible narrative."""
        narrative = (
            "*Mira tritt auf den Pfad und sagt:* 'Ich bin schon "
            "unterwegs zum Bach, ihr könnt mir folgen.'"
        )
        assert _looks_imperative_template(narrative) is False

    def test_narrative_with_reagiert_auf_einen_not_flagged(self) -> None:
        """'reagiert auf einen lauten Knall' is valid narrative —
        the v5 heuristic only flags the FULL phrase 'reagiert auf
        einen anderen charakter' (verbatim template marker)."""
        narrative = (
            "*Yara reagiert auf einen lauten Knall hinter ihr und "
            "fährt zusammen* 'Was war das?'"
        )
        assert _looks_imperative_template(narrative) is False

    def test_narrative_with_macht_eine_konkrete_aktion_full_phrase(self) -> None:
        """The full template phrase 'macht eine konkrete aktion'
        IS still flagged (it's verbatim from Sandra's template).
        Only the looser 'macht eine konkrete' (without 'aktion')
        was the false-positive risk; we kept the precise variant."""
        partial = (
            "Sandra macht eine konkrete Aktion statt nur zu fühlen."
        )
        assert _looks_imperative_template(partial) is True

    def test_real_castaway_yara_leak_caught(self) -> None:
        """Verbatim from Mike's report — a full leak of the Yara
        prompt as displayed pulse text. The full thing must be
        flagged, not just a fragment."""
        leaked = (
            "Yara macht etwas BEOBACHTBARES. Wähle EINS: zeigt mit "
            "dem Finger auf etwas Konkretes (Rauch, Trümmer im "
            "Wasser, Spur im Sand, ein Vogel über dem Hügel), fasst "
            "die Lage in EINEM Satz zusammen ('Wir haben Wasser, "
            "aber kein Werkzeug.'), oder stellt eine konkrete Frage "
            "die voranbringt ('Wer hat das Süßwasser schon "
            "getestet?'). KEIN SCHWEIGEN — wenn keiner antwortet, "
            "spricht sie aus was sie gerade sieht. Verbote: KEIN "
            "passives Sitzen, KEIN 'starre in den Sand'-Loop."
        )
        assert _looks_imperative_template(leaked) is True

    def test_real_castaway_lena_leak_caught(self) -> None:
        leaked = (
            "Lena REAGIERT auf einen anderen Charakter NAMENTLICH. "
            "Wähle EINS: spricht Sandra direkt an ('Sandra, ich hab "
            "Hunger'), folgt Mira zum Wald, fragt Yara was zu tun "
            "ist, klammert sich an Sandras Arm. Sie ist 16 und "
            "ängstlich, ABER KEINE STATUE — sie macht etwas, sie "
            "spricht jemanden an. Verbote: KEIN passives "
            "Sand-Anstarren, KEIN 'mein Kopf dröhnt'-Loop. Sie "
            "nutzt einen Namen einer anderen Person."
        )
        assert _looks_imperative_template(leaked) is True


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


# ─── Phase 13.5 hotfix v4: imperative-prompt sidestep ──────────────


class TestImperativeSidestep:
    """Phase 13.5 hotfix v4 — when card.proactive_pulse_prompt looks
    imperative, the plugin must NOT pass it as style_guidance to the
    generator. Mike's chat showed Yara's pulse as the generic default
    text every single time because e4b kept echoing the imperative
    instead of generating narrative; my heuristic caught the echo,
    fallback fired, and the default leaked. Stripping the imperative
    from the guidance lets the generator produce persona-driven
    narrative reliably."""

    def test_imperative_prompt_not_passed_as_guidance(self) -> None:
        """Simulate the plugin's branch: when raw guidance looks
        imperative, the generator gets called WITHOUT guidance."""
        from plugins.character_chat.character_chat_plugin import (
            _looks_imperative_template,
        )

        imperative = (
            "Yara macht etwas BEOBACHTBARES. Wähle EINS: zeigt mit "
            "dem Finger... Verbote: KEIN passives Sitzen."
        )
        # Confirm the heuristic flags it (consistency with prior tests).
        assert _looks_imperative_template(imperative) is True

        llm = _FakeLLM()
        gen = PulseGenerator(llm_chat=llm, brain="e4b")
        card = _card("Yara", pulse_prompt=imperative)
        # The plugin calls generate with style_guidance="" when the
        # raw prompt is imperative. Verify the prompt-sans-guidance
        # path works as expected.
        asyncio.run(gen.generate(
            character=card,
            others_in_session=[],
            recent_history=[],
            style_guidance="",  # plugin clears imperatives before pass-through
        ))
        user_prompt = llm.calls[0]["messages"][1]["content"]
        # Aktions-Vorgabe section not present (no guidance to embed).
        assert "Aktions-Vorgabe" not in user_prompt
        # The imperative text itself doesn't appear in the prompt
        # body either — the LLM has nothing to echo.
        assert "Wähle EINS" not in user_prompt
        assert "Verbote:" not in user_prompt

    def test_narrative_prompt_still_passed_as_guidance(self) -> None:
        """A hand-written narrative pulse_prompt (the original design
        intent) still flows through to the generator as guidance —
        only imperatives get stripped."""
        narrative = (
            "*hört ein Knirschen aus dem Wald und zuckt unsicher "
            "zusammen, Augen weit aufgerissen*"
        )
        from plugins.character_chat.character_chat_plugin import (
            _looks_imperative_template,
        )
        assert _looks_imperative_template(narrative) is False
        # In the plugin path, this narrative would flow through.
        llm = _FakeLLM()
        gen = PulseGenerator(llm_chat=llm, brain="e4b")
        card = _card("Yara", pulse_prompt=narrative)
        asyncio.run(gen.generate(
            character=card,
            others_in_session=[],
            recent_history=[],
            style_guidance=narrative,  # narrative passes through
        ))
        user_prompt = llm.calls[0]["messages"][1]["content"]
        assert "Aktions-Vorgabe" in user_prompt
        assert "Knirschen aus dem Wald" in user_prompt
