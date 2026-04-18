"""
Lexy AI - Expert Panel Synthesizer.

Generates a final structured synthesis from all panel discussion messages.
Produces consensus points, dissent points, action items, and an overall summary.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.plugin_system.plugin_api import PluginAPI

log = get_logger(module="expert_panel.synthesizer")

_SYNTHESIS_SYSTEM_PROMPT: str = (
    "Du bist der Synthesizer eines Expertenpanels. Deine Aufgabe ist es, "
    "die gesamte Diskussion in ein strukturiertes Ergebnis zusammenzufassen.\n\n"
    "Antworte AUSSCHLIESSLICH als JSON-Objekt mit diesem Format:\n"
    "{\n"
    '  "summary": "Zusammenfassung der Diskussion in 3-5 Saetzen",\n'
    '  "consensus_points": [\n'
    '    "Punkt, bei dem sich die Experten einig sind"\n'
    "  ],\n"
    '  "dissent_points": [\n'
    '    "Punkt, bei dem Uneinigkeit herrscht"\n'
    "  ],\n"
    '  "action_items": [\n'
    '    "Konkreter naechster Schritt"\n'
    "  ]\n"
    "}\n\n"
    "Regeln:\n"
    "- Die Zusammenfassung soll praegnant und neutral sein.\n"
    "- Consensus: nur echte Uebereinstimmungen (mindestens 2 Experten).\n"
    "- Dissent: wichtige offene Streitpunkte.\n"
    "- Action Items: konkrete, umsetzbare naechste Schritte.\n"
    "- Maximal 5 Punkte pro Kategorie.\n"
    "- Antworte NUR mit dem JSON, kein anderer Text."
)


class PanelSynthesizer:
    """Generates a structured final synthesis from panel discussion messages."""

    async def synthesize(
        self,
        messages: list[dict[str, Any]],
        topic: str,
        api: "PluginAPI",
        brain: str = "e4b",
    ) -> dict[str, Any]:
        """
        Generate final synthesis from all panel messages.

        Args:
            messages: All panel messages as dicts (role, phase, round, content).
            topic: The discussion topic.
            api: PluginAPI for LLM access.
            brain: Which brain to use for synthesis.

        Returns:
            {summary: str, consensus_points: [str], dissent_points: [str], action_items: [str]}
        """
        if not messages:
            return {
                "summary": "Keine Diskussion stattgefunden.",
                "consensus_points": [],
                "dissent_points": [],
                "action_items": [],
            }

        # Baue den vollstaendigen Diskussions-Kontext
        context_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            phase = msg.get("phase", "")
            round_num = msg.get("round", 0)
            content = msg.get("content", "")
            context_parts.append(
                f"[{role} | Phase: {phase} | Runde {round_num}]\n{content}"
            )
        discussion_text = "\n\n---\n\n".join(context_parts)

        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Thema: {topic}\n\n"
                    f"Vollstaendige Diskussion:\n{discussion_text}\n\n"
                    "Erstelle die Synthese:"
                ),
            },
        ]

        try:
            raw = await api.llm_chat(
                llm_messages, brain=brain, max_tokens=800, temperature=0.4
            )
        except Exception as exc:  # noqa: BLE001
            log.error("synthesizer.llm_failed", error=str(exc))
            return {
                "summary": f"Synthese fehlgeschlagen: {exc}",
                "consensus_points": [],
                "dissent_points": [],
                "action_items": [],
            }

        result = self._parse_synthesis(raw)
        log.info(
            "synthesizer.done",
            consensus=len(result["consensus_points"]),
            dissent=len(result["dissent_points"]),
            actions=len(result["action_items"]),
        )
        return result

    @staticmethod
    def _parse_synthesis(raw: str) -> dict[str, Any]:
        """
        Parse structured JSON from the LLM synthesis response.

        Tolerant: handles markdown fences and extra surrounding text.
        """
        text = raw.strip()

        # Entferne Markdown Code-Fences falls vorhanden
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                stripped = part.strip()
                if stripped.startswith("json"):
                    stripped = stripped[4:].strip()
                if stripped.startswith("{"):
                    text = stripped
                    break

        # Finde das JSON-Objekt
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            log.warning(
                "synthesizer.parse_failed", reason="no_json_found", raw=text[:200]
            )
            return {
                "summary": text[:500] if text else "Synthese konnte nicht geparst werden.",
                "consensus_points": [],
                "dissent_points": [],
                "action_items": [],
            }

        json_str = text[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            log.warning(
                "synthesizer.parse_failed", reason="invalid_json", error=str(exc)
            )
            return {
                "summary": text[:500],
                "consensus_points": [],
                "dissent_points": [],
                "action_items": [],
            }

        # Validiere und extrahiere mit Fallbacks
        summary = data.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)

        consensus = data.get("consensus_points", [])
        if not isinstance(consensus, list):
            consensus = []
        consensus = [str(p) for p in consensus if p][:5]

        dissent = data.get("dissent_points", [])
        if not isinstance(dissent, list):
            dissent = []
        dissent = [str(p) for p in dissent if p][:5]

        actions = data.get("action_items", [])
        if not isinstance(actions, list):
            actions = []
        actions = [str(a) for a in actions if a][:5]

        return {
            "summary": summary,
            "consensus_points": consensus,
            "dissent_points": dissent,
            "action_items": actions,
        }
