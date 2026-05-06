"""
LLM-driven pulse-text generator for character_chat.

The original pulse pipeline picks one of five hand-written sentences from
``_DEFAULT_PULSES`` (per age stage). That makes proactive characters feel
puppet-like: every adult pulse is "*sieht auf und sucht Blickkontakt*", every
toddler pulse is "*zieht an Mamas Ärmel*". This module replaces that static
table with a small LLM call (E4B by default) that sees the character's
persona, current state, the other characters in the session, and the recent
chat history — then generates a single plausible next action OR a question
to another character.

The class is deliberately decoupled from the plugin: it takes a generic
``llm_chat`` callable so tests can inject a fake without booting the full
``LexyApp``. It only generates the *text* — registering a pulse round and
broadcasting it stays inside the plugin.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable

from .character_card import CharacterCard, _format_state_block


log = logging.getLogger(__name__)


LLMChat = Callable[..., Awaitable[str]]


class PulseGenerator:
    """Generate a one-shot pulse text for a proactive character."""

    def __init__(
        self,
        *,
        llm_chat: LLMChat,
        brain: str = "e4b",
        max_tokens: int = 200,
        temperature: float = 0.85,
        history_window: int = 6,
    ) -> None:
        self._llm_chat = llm_chat
        self._brain = brain
        self._max_tokens = max(40, int(max_tokens))
        self._temperature = float(temperature)
        self._history_window = max(0, int(history_window))

    async def generate(
        self,
        *,
        character: CharacterCard,
        others_in_session: list[CharacterCard],
        recent_history: list[dict[str, Any]],
        scene: str = "",
    ) -> str:
        """Return a single-sentence-or-two pulse text for ``character``.

        Returns ``""`` if the LLM call fails or produces nothing usable —
        the caller falls back to ``character.proactive_pulse_prompt`` or
        the static age-stage default in that case.
        """
        history_blurb = _format_history_tail(
            recent_history, limit=self._history_window
        )
        others_blurb = _format_others(others_in_session, exclude_id=character.id)
        state_blurb = _format_state_block(character.state or {})
        persona = (character.persona or "").strip() or f"(age_stage={character.age_stage})"

        # System: who they are + how to behave.
        system = (
            f"Du bist {character.name} und entscheidest, was du JETZT von "
            "dir aus tust oder sagst, ohne dass jemand dich angesprochen hat. "
            "Generiere genau EINE knappe Aktion ODER eine Frage an einen der "
            "anderen anwesenden Charaktere. Format: Action in *Sternchen* "
            "und/oder Dialog, 1-2 Sätze. Keine Meta-Kommentare, keine "
            "Begründungen, keine Anführungszeichen um die ganze Antwort. "
            "Wenn du jemanden ansprichst, NENN diese Person beim Namen "
            "(\"Drell, hast du ...?\") — nicht @-Syntax."
        )

        user_parts: list[str] = []
        user_parts.append(f"## Deine Persona\n{persona}")
        if state_blurb:
            user_parts.append(f"## Dein aktueller Zustand\n{state_blurb}")
        if others_blurb:
            user_parts.append(f"## Andere anwesende Charaktere\n{others_blurb}")
        if scene.strip():
            user_parts.append(f"## Szene\n{scene.strip()}")
        if history_blurb:
            user_parts.append(f"## Letzte Chat-Zeilen\n{history_blurb}")
        user_parts.append(
            "## Was tust du jetzt?\n"
            "Eine plausible Folgeaktion oder eine Frage an genau eine der "
            "anderen Personen. Sei konkret (z.B. 'geht zur Kaffeemaschine', "
            "'fragt Drell ob er den Antrieb gecheckt hat'). KEINE inneren "
            "Monologe, KEINE Sätze die mit 'Ich denke' beginnen."
        )

        try:
            raw = await self._llm_chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": "\n\n".join(user_parts)},
                ],
                brain=self._brain,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "character_chat.pulse_generation_failed character=%s error=%s",
                character.name,
                exc,
            )
            return ""

        text = _post_process(raw or "")
        if text:
            log.info(
                "character_chat.pulse_generated character=%s brain=%s len=%d",
                character.name,
                self._brain,
                len(text),
            )
        return text


# ─── Helpers ─────────────────────────────────────────────────────────────


def _format_history_tail(
    history: list[dict[str, Any]], *, limit: int
) -> str:
    """Compact rendering of the last ``limit`` history entries."""
    if not history or limit <= 0:
        return ""
    tail = history[-limit:]
    lines: list[str] = []
    for entry in tail:
        role = str(entry.get("role", "user"))
        name = str(entry.get("name") or ("Du" if role == "user" else "Lexy"))
        content = str(entry.get("content", "")).strip()
        if not content:
            continue
        if len(content) > 200:
            content = content[:200].rstrip() + "…"
        prefix = f"{name}:"
        lines.append(f"{prefix} {content}")
    return "\n".join(lines)


def _format_others(
    others: list[CharacterCard], *, exclude_id: str
) -> str:
    """List the other present characters with one-line persona excerpts."""
    bits: list[str] = []
    for c in others:
        if c.id == exclude_id:
            continue
        excerpt = (c.persona or "").strip().split("\n", 1)[0]
        if len(excerpt) > 100:
            excerpt = excerpt[:100].rstrip() + "…"
        if excerpt:
            bits.append(f"- {c.name}: {excerpt}")
        else:
            bits.append(f"- {c.name}")
    return "\n".join(bits)


_QUOTE_WRAPPED_RE = re.compile(r'^[\'"](.+)[\'"]$', re.DOTALL)


def _post_process(raw: str) -> str:
    """Clean up common LLM tics in the pulse output.

    Strips:
    * Leading/trailing whitespace.
    * Matched outer quotes (some models wrap their reply in "...").
    * Leading character-name labels ("Mara: " when it's already obvious it's
      Mara speaking — the pulse pipeline prepends the speaker name itself).
    * ``[PASS]`` markers — a pulse-generating character cannot pass; if the
      LLM produces one, we treat it as no pulse and let the fallback fire.
    * Any ``<state>...</state>`` block — pulses don't update state and the
      block would leak into the visible pulse text.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    # Strip <state>...</state> FIRST so a trailing block doesn't break the
    # outer-quote heuristic below ("..." <state>...</state> wouldn't end in
    # a quote and we'd miss the unwrap).
    text = re.sub(
        r"<state>.*?</state>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Drop optional outer quotes.
    m = _QUOTE_WRAPPED_RE.match(text)
    if m:
        text = m.group(1).strip()

    # Drop a leading "Name:" label up to ~30 chars.
    text = re.sub(r"^[^:\n]{1,30}:\s*", "", text, count=1)

    if not text or text.upper() in ("[PASS]", "[SILENT]", "PASS"):
        return ""

    # Hard cap so a runaway generation doesn't dump 800 tokens into the pulse.
    if len(text) > 600:
        text = text[:600].rstrip() + "…"
    return text


__all__ = ["PulseGenerator"]
