"""
Lexy AI - Coordination: FactExtractor.

A generic "read text → structured facts (JSON)" pass, same shape as
:class:`Referee` / :class:`ConvergenceDetector` (one cheap LLM call,
tolerant JSON parse, fail-safe to ``{}``).

Used by the RP physical-continuity tracker: after a round it reads the
narration and answers "where is the baby now / who holds it", so the next
turn's prompt states the current physical reality and the character can't
contradict it ("puts the baby down" → next message it's somehow back on
her arm). Generic enough for any "extract these fields" need.
"""

from __future__ import annotations

import json
from typing import Any

from lexy_core.coordination.convergence import LLMChat
from lexy_core.utils.logging import get_logger

log = get_logger(module="coordination.fact_extractor")

_SYSTEM_PROMPT: str = (
    "Du extrahierst aktuelle Fakten aus einem Rollenspiel-Text. Lies den "
    "Text und gib NUR die nach der Anweisung gefragten Fakten als JSON-"
    "Objekt zurueck. Wenn ein Fakt im Text nicht eindeutig vorkommt, lass "
    "ihn WEG (rate nicht). Antworte ausschliesslich mit dem JSON-Objekt, "
    "kein weiterer Text."
)


class FactExtractor:
    """Extracts a small structured fact dict from free text via one LLM call."""

    async def extract(
        self,
        text: str,
        instruction: str,
        llm_chat: LLMChat,
        brain: str = "e4b",
        max_tokens: int = 300,
    ) -> dict[str, Any]:
        """Return facts parsed from ``text`` per ``instruction``.

        Args:
            text: The narration / source text to read.
            instruction: What to extract + the expected JSON shape.
            llm_chat: Async ``(messages, brain, max_tokens, temperature) -> str``.
            brain: Cheap brain by default.

        Returns ``{}`` on empty input or any LLM/parse error (fail-safe — a
        missed extraction just leaves the existing facts unchanged).
        """
        if not (text or "").strip():
            return {}
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\nText:\n{text}"},
        ]
        try:
            raw = await llm_chat(
                messages, brain=brain, max_tokens=max_tokens, temperature=0.1
            )
        except Exception as exc:  # noqa: BLE001
            log.error("fact_extractor.llm_failed", error=str(exc))
            return {}
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
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
            return {}
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            log.warning("fact_extractor.parse_failed", raw=text[:200])
            return {}
        return data if isinstance(data, dict) else {}
