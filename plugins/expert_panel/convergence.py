"""
Lexy AI - Expert Panel Convergence Detector.

Analyzes discussion messages from the latest round to identify agreement
points across panel roles. Uses an LLM call to parse the natural-language
discussion and extract structured agreement data.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.plugin_system.plugin_api import PluginAPI

log = get_logger(module="expert_panel.convergence")

_CONVERGENCE_SYSTEM_PROMPT: str = (
    "Du bist ein Diskussions-Analyst. Analysiere die Beitraege der Experten "
    "und identifiziere Punkte, bei denen mehrere Experten uebereinstimmen.\n\n"
    "Antworte AUSSCHLIESSLICH als JSON-Objekt mit diesem Format:\n"
    "{\n"
    '  "agreements": [\n'
    '    {"point": "Beschreibung des Konsens-Punkts", '
    '"agreeing_roles": ["rolle1", "rolle2"]}\n'
    "  ]\n"
    "}\n\n"
    "Regeln:\n"
    "- Nur echte inhaltliche Uebereinstimmungen zaehlen.\n"
    "- Mindestens 2 Rollen muessen zustimmen.\n"
    "- Maximal 5 Agreement-Punkte.\n"
    "- Antworte NUR mit dem JSON, kein anderer Text."
)


class ConvergenceDetector:
    """Detects agreement across panel agent contributions."""

    async def check(
        self,
        messages: list[dict[str, Any]],
        roles: list[str],
        threshold: int,
        api: "PluginAPI",
        brain: str = "e4b",
    ) -> dict[str, Any]:
        """
        Analyze discussion messages and count agreements.

        Args:
            messages: All panel messages as dicts (role, phase, round, content).
            roles: List of participating roles.
            threshold: Number of agreement points required for convergence.
            api: PluginAPI for LLM access.
            brain: Which brain to use for analysis.

        Returns:
            {converged: bool, agreements: [{point, agreeing_roles}], agreement_count: int}
        """
        if not messages:
            return {"converged": False, "agreements": [], "agreement_count": 0}

        # Baue den Diskussions-Kontext aus allen Nachrichten
        context_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            phase = msg.get("phase", "")
            round_num = msg.get("round", 0)
            context_parts.append(
                f"[{role} | Phase: {phase} | Runde {round_num}]\n{content}"
            )
        discussion_text = "\n\n---\n\n".join(context_parts)

        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": _CONVERGENCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Teilnehmende Rollen: {', '.join(roles)}\n\n"
                    f"Diskussion:\n{discussion_text}\n\n"
                    "Identifiziere die Uebereinstimmungen:"
                ),
            },
        ]

        try:
            raw = await api.llm_chat(
                llm_messages, brain=brain, max_tokens=500, temperature=0.3
            )
        except Exception as exc:  # noqa: BLE001
            log.error("convergence.llm_failed", error=str(exc))
            return {"converged": False, "agreements": [], "agreement_count": 0}

        agreements = self._parse_agreements(raw, roles)
        agreement_count = len(agreements)
        converged = agreement_count >= threshold

        log.info(
            "convergence.checked",
            agreement_count=agreement_count,
            threshold=threshold,
            converged=converged,
        )

        return {
            "converged": converged,
            "agreements": agreements,
            "agreement_count": agreement_count,
        }

    @staticmethod
    def _parse_agreements(
        raw: str, valid_roles: list[str]
    ) -> list[dict[str, Any]]:
        """
        Parse structured JSON from the LLM response.

        Tolerant: extracts the JSON object even if surrounded by markdown
        fences or extra text.
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
            log.warning("convergence.parse_failed", reason="no_json_found", raw=text[:200])
            return []

        json_str = text[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            log.warning("convergence.parse_failed", reason="invalid_json", error=str(exc))
            return []

        raw_agreements = data.get("agreements", [])
        if not isinstance(raw_agreements, list):
            return []

        # Validiere und bereinige
        result: list[dict[str, Any]] = []
        for item in raw_agreements:
            if not isinstance(item, dict):
                continue
            point = item.get("point", "")
            agreeing = item.get("agreeing_roles", [])
            if not point or not isinstance(agreeing, list):
                continue
            # Nur valide Rollen behalten
            valid = [r for r in agreeing if r in valid_roles]
            if len(valid) >= 2:
                result.append({"point": point, "agreeing_roles": valid})

        return result[:5]  # Maximal 5 Punkte
