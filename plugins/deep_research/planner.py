"""
Topic → Sub-queries planner.

Single LLM call against the configured ``planner_brain`` (default a4b).
The model is asked to decompose the user's topic into a small number of
search-friendly sub-queries that, taken together, cover the topic.

Output convention: a JSON array of strings. Tolerant to common LLM
deviations (numbered list, code-fenced JSON, leading prose) — the
parser accepts any of those.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable


log = logging.getLogger(__name__)


LLMChat = Callable[..., Awaitable[str]]


# Minimum characters in a usable sub-query — anything shorter is
# probably an LLM filler word and gets dropped.
_MIN_QUERY_LEN: int = 4


class Planner:
    """Decomposes a research topic into search-friendly sub-queries."""

    _SYSTEM_PROMPT = (
        "Du bist Lexys Recherche-Planer. Eine User-Frage zu einem Thema "
        "soll in {min}-{max} konkrete Such-Anfragen zerlegt werden, die "
        "zusammen das Thema gut abdecken. Jede Anfrage soll:\n"
        "- eigenständig googelbar sein (kein Pronomen, kein 'es', "
        "kein 'das', kein 'sie')\n"
        "- einen anderen Aspekt des Themas abdecken\n"
        "- Stichworte enthalten, die ein Suchindex versteht\n"
        "- in der vom User verwendeten Sprache formuliert sein "
        "(default Deutsch)\n\n"
        "Antworte AUSSCHLIESSLICH als JSON-Array von Strings. KEIN "
        "Markdown, keine Erklärung, keine Code-Fence. Beispiel:\n"
        '["Tesla Cybertruck Spezifikationen", "Cybertruck Lieferprobleme '
        '2024", "Cybertruck Reichweite Tests"]'
    )

    def __init__(
        self,
        *,
        llm_chat: LLMChat,
        brain: str = "a4b",
        max_tokens: int = 600,
        temperature: float = 0.4,
        min_subqueries: int = 3,
        max_subqueries: int = 7,
        default_subqueries: int = 5,
    ) -> None:
        self._chat = llm_chat
        self._brain = brain
        self._max_tokens = max(120, int(max_tokens))
        self._temperature = float(temperature)
        self._min = max(1, int(min_subqueries))
        self._max = max(self._min, int(max_subqueries))
        self._default = max(self._min, min(int(default_subqueries), self._max))

    async def plan(
        self,
        *,
        topic: str,
        language: str = "de",
        target_count: int | None = None,
    ) -> list[str]:
        """Return a list of sub-queries. Falls back to ``[topic]`` on
        unrecoverable parsing failures so the pipeline never stalls."""
        target = target_count if target_count is not None else self._default
        target = max(self._min, min(int(target), self._max))

        system = self._SYSTEM_PROMPT.format(
            min=self._min, max=self._max,
        )
        user = (
            f"Sprache: {language}\n"
            f"Ziel-Anzahl: ungefähr {target}\n"
            f"Thema: {topic.strip()}\n\n"
            "Liefere das JSON-Array."
        )
        try:
            raw = await self._chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                brain=self._brain,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("deep_research.plan_llm_failed err=%s", exc)
            return [topic.strip()]

        queries = parse_subqueries(raw)
        # Best-effort cleanup: dedupe (case-insensitive), drop short
        # filler, cap at max_subqueries.
        seen: set[str] = set()
        clean: list[str] = []
        for q in queries:
            q = q.strip()
            if len(q) < _MIN_QUERY_LEN:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(q)
        if not clean:
            return [topic.strip()]
        return clean[: self._max]


# ─── Parser ─────────────────────────────────────────────────────────


def parse_subqueries(raw: str) -> list[str]:
    """Best-effort extraction of sub-queries from any LLM blob.

    Accepts:

    * A clean JSON array of strings.
    * JSON wrapped in ```json fences.
    * A numbered list (``1. foo\\n2. bar``).
    * A bullet list (``- foo\\n- bar``).
    * Free-text lines, falling back as the last resort.
    """
    text = (raw or "").strip()
    if not text:
        return []

    # Strip a leading code-fence (```json or ```).
    if text.startswith("```"):
        first_break = text.find("\n")
        if first_break != -1:
            text = text[first_break + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Strict JSON path.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try locating the first JSON-array substring.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                cleaned = [str(x).strip() for x in parsed if str(x).strip()]
                if cleaned:
                    return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

    # Numbered / bullet lists.
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1." / "1)" / "-" / "*" / "•"
        line = re.sub(r"^\s*(?:\d+[\.\)]\s+|[-*•]\s+)", "", line).strip()
        # Drop surrounding quotes if any (LLMs sometimes wrap each item).
        if (line.startswith('"') and line.endswith('"')) or \
           (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        # Drop trailing comma (JSON-fragment leftover).
        if line.endswith(","):
            line = line[:-1].rstrip()
        if line:
            out.append(line)
    return out
