"""
Lexy AI - Coordination: Referee (the game master).

The piece that turns narration back into simulation. Characters are driven
by dynamic situational prompts and reply with free, immersive prose — no
machine syntax in the roleplay. But prose alone changes no state: when the
mother *says* "I feed the baby", nothing knows whether ``baby.hunger``
should drop.

Like a pen-&-paper game master, the :class:`Referee` adjudicates: it reads
the open :class:`Demand` plus the character's narration and decides whether
the demand was actually satisfied. Only a concrete, goal-directed action
counts — merely commenting or "taking note" does not. The loop then applies
the verdict to the :class:`WorldState` (satisfied → the number drops;
not satisfied → the demand stays open and escalates next tick).

Implementation mirrors :class:`ConvergenceDetector`: one cheap LLM call,
tolerant JSON parsing, and a **fail-safe default of NOT satisfied** so an
error or an unsure model lets consequences escalate rather than silently
clearing an obligation.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from lexy_core.coordination.convergence import LLMChat
from lexy_core.coordination.world_state import Demand
from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.referee")

_REFEREE_SYSTEM_PROMPT: str = (
    "Du bist der Spielleiter (Schiedsrichter) einer Rollenspiel-Simulation. "
    "Eine Figur stand vor einer konkreten Anforderung. Lies, was die Figur "
    "tatsaechlich getan bzw. erzaehlt hat, und entscheide, ob die Anforderung "
    "dadurch erfuellt wurde.\n\n"
    "WICHTIG: Reines Kommentieren, Bemerken oder Zur-Kenntnis-Nehmen erfuellt "
    "die Anforderung NICHT. Nur eine konkrete, zielgerichtete Handlung zaehlt.\n\n"
    "Antworte AUSSCHLIESSLICH als JSON-Objekt:\n"
    '{"satisfied": true, "magnitude": 0.0, "rationale": "kurze Begruendung"}\n\n'
    "Regeln:\n"
    "- satisfied=true NUR bei einer konkreten Handlung, die die Anforderung adressiert.\n"
    "- magnitude (0.0-1.0): wie vollstaendig erfuellt (1.0 = voll, 0.3 = teilweise).\n"
    "- Im Zweifel satisfied=false.\n"
    "- Antworte NUR mit dem JSON, kein anderer Text."
)


class Verdict(BaseModel):
    """The referee's ruling on whether a demand was satisfied."""

    satisfied: bool = False
    magnitude: float = 0.0      # 0.0..1.0 — how fully the demand was met
    rationale: str = ""


class Referee:
    """Adjudicates whether a narrated action satisfied an open demand."""

    async def adjudicate(
        self,
        demand: Demand,
        narration: str,
        llm_chat: LLMChat,
        brain: str = "e4b",
    ) -> Verdict:
        """Rule on ``narration`` against ``demand``.

        Args:
            demand: The open obligation (need, entity, attribute, urgency).
            narration: What the character actually said/did this turn.
            llm_chat: Async ``(messages, brain, max_tokens, temperature) -> str``.
            brain: Which brain to use (a cheap one by default).

        Returns:
            A :class:`Verdict`. On empty narration or any LLM/parse error the
            verdict is **not satisfied** (fail-safe → consequences escalate).
        """
        if not (narration or "").strip():
            return Verdict(rationale="leere Narration")

        messages = [
            {"role": "system", "content": _REFEREE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Anforderung: {demand.need} "
                    f"(betrifft {demand.entity}.{demand.attribute}, "
                    f"Dringlichkeit {demand.urgency}).\n\n"
                    f"Was die Figur tat/erzaehlte:\n{narration}\n\n"
                    "Wurde die Anforderung erfuellt?"
                ),
            },
        ]

        try:
            raw = await llm_chat(messages, brain=brain, max_tokens=200, temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            log.error("referee.llm_failed", error=str(exc), need=demand.need)
            return Verdict(rationale=f"llm_failed: {exc}")

        verdict = self._parse_verdict(raw)
        log.info(
            "referee.adjudicated",
            need=demand.need,
            entity=demand.entity,
            satisfied=verdict.satisfied,
            magnitude=round(verdict.magnitude, 2),
        )
        return verdict

    @staticmethod
    def _parse_verdict(raw: str) -> Verdict:
        """Parse the JSON ruling tolerantly. Fail-safe to NOT satisfied."""
        text = (raw or "").strip()

        if "```" in text:
            for part in text.split("```"):
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("{"):
                    text = stripped
                    break

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning("referee.parse_failed", reason="no_json", raw=text[:200])
            return Verdict(rationale="kein JSON")

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            log.warning("referee.parse_failed", reason="invalid_json", error=str(exc))
            return Verdict(rationale="ungueltiges JSON")

        satisfied = bool(data.get("satisfied", False))
        try:
            magnitude = float(data.get("magnitude", 0.0))
        except (TypeError, ValueError):
            magnitude = 0.0
        magnitude = max(0.0, min(1.0, magnitude))
        rationale = str(data.get("rationale", ""))[:240]

        # A "satisfied" verdict with zero magnitude is contradictory — treat
        # it as a minimal real effect so the loop makes at least some progress.
        if satisfied and magnitude == 0.0:
            magnitude = 1.0

        return Verdict(satisfied=satisfied, magnitude=magnitude, rationale=rationale)
