"""
Lexy AI - Coordination: Convergence Detector.

Generalised from ``plugins/expert_panel/convergence.py``. Given the
contributions on a shared board, one LLM call extracts the points where
several participants agree. A deliberation loop uses this to decide when a
discussion has converged and can stop — instead of running a fixed number
of rounds, or (the failure mode) everyone politely nodding forever.

Decoupled from :class:`PluginAPI`: callers pass an async ``llm_chat``
callable, so this is trivially unit-testable with a stub and reusable by
the expert panel, the orchestrator's council mode, and any future
deliberation feature.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.convergence")

# An async callable ``(messages, brain, max_tokens, temperature) -> str``.
# Matches ``PluginAPI.llm_chat`` so plugins can pass it directly.
LLMChat = Callable[..., Awaitable[str]]

_CONVERGENCE_SYSTEM_PROMPT: str = (
    "Du bist ein Diskussions-Analyst. Analysiere die Beitraege der "
    "Teilnehmer und identifiziere Punkte, bei denen mehrere Teilnehmer "
    "uebereinstimmen.\n\n"
    "Antworte AUSSCHLIESSLICH als JSON-Objekt mit diesem Format:\n"
    "{\n"
    '  "agreements": [\n'
    '    {"point": "Beschreibung des Konsens-Punkts", '
    '"agreeing": ["teilnehmer1", "teilnehmer2"]}\n'
    "  ]\n"
    "}\n\n"
    "Regeln:\n"
    "- Nur echte inhaltliche Uebereinstimmungen zaehlen.\n"
    "- Mindestens 2 Teilnehmer muessen zustimmen.\n"
    "- Maximal 5 Agreement-Punkte.\n"
    "- Antworte NUR mit dem JSON, kein anderer Text."
)


class ConvergenceResult(BaseModel):
    """Outcome of a convergence check."""

    converged: bool = False
    agreements: list[dict[str, Any]] = Field(default_factory=list)
    agreement_count: int = 0


class ConvergenceDetector:
    """Detects agreement across contributions on a shared board."""

    async def check(
        self,
        contributions: list[dict[str, Any]],
        participants: list[str],
        threshold: int,
        llm_chat: LLMChat,
        brain: str = "e4b",
    ) -> ConvergenceResult:
        """Analyse contributions and decide whether the discussion converged.

        Args:
            contributions: Board posts/messages as dicts. Each should carry a
                ``content`` (or ``body``) and an author identifier under
                ``author`` or ``role``; optional ``phase`` / ``round``.
            participants: Valid participant identifiers; an agreement only
                counts when ≥2 of these agree.
            threshold: Number of agreement points required to declare
                convergence.
            llm_chat: Async ``(messages, brain, max_tokens, temperature) -> str``.
            brain: Which brain to use for the analysis (cheap one by default).

        Returns:
            A :class:`ConvergenceResult`.
        """
        if not contributions:
            return ConvergenceResult()

        context_parts: list[str] = []
        for item in contributions:
            author = item.get("author") or item.get("role") or "unknown"
            content = item.get("content") or item.get("body") or ""
            phase = item.get("phase", "")
            round_num = item.get("round", item.get("round_num", 0))
            header = f"[{author}"
            if phase:
                header += f" | Phase: {phase}"
            if round_num:
                header += f" | Runde {round_num}"
            header += "]"
            context_parts.append(f"{header}\n{content}")
        discussion_text = "\n\n---\n\n".join(context_parts)

        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": _CONVERGENCE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Teilnehmer: {', '.join(participants)}\n\n"
                    f"Diskussion:\n{discussion_text}\n\n"
                    "Identifiziere die Uebereinstimmungen:"
                ),
            },
        ]

        try:
            raw = await llm_chat(
                llm_messages, brain=brain, max_tokens=500, temperature=0.3
            )
        except Exception as exc:  # noqa: BLE001
            log.error("convergence.llm_failed", error=str(exc))
            return ConvergenceResult()

        agreements = self._parse_agreements(raw, participants)
        agreement_count = len(agreements)
        converged = agreement_count >= threshold

        log.info(
            "convergence.checked",
            agreement_count=agreement_count,
            threshold=threshold,
            converged=converged,
        )
        return ConvergenceResult(
            converged=converged,
            agreements=agreements,
            agreement_count=agreement_count,
        )

    @staticmethod
    def _parse_agreements(
        raw: str, valid_participants: list[str]
    ) -> list[dict[str, Any]]:
        """Parse the structured JSON from the LLM response.

        Tolerant: extracts the JSON object even if wrapped in markdown
        fences or surrounded by stray text. Accepts both ``agreeing`` and
        ``agreeing_roles`` keys (the latter for backwards-compat with the
        original expert_panel prompt). Returns at most 5 validated points.
        """
        text = (raw or "").strip()

        # Strip markdown code fences if present.
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
            log.warning("convergence.parse_failed", reason="no_json", raw=text[:200])
            return []

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError as exc:
            log.warning("convergence.parse_failed", reason="invalid_json", error=str(exc))
            return []

        raw_agreements = data.get("agreements", [])
        if not isinstance(raw_agreements, list):
            return []

        result: list[dict[str, Any]] = []
        for item in raw_agreements:
            if not isinstance(item, dict):
                continue
            point = item.get("point", "")
            agreeing = item.get("agreeing")
            if agreeing is None:
                agreeing = item.get("agreeing_roles", [])
            if not point or not isinstance(agreeing, list):
                continue
            valid = [a for a in agreeing if a in valid_participants]
            if len(valid) >= 2:
                result.append({"point": point, "agreeing": valid})

        return result[:5]
