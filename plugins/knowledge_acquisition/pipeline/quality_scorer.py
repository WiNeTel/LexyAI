"""
Lexy AI - Knowledge Acquisition: Quality Scorer.

LLM-based content quality rating (1-5).  Uses the fast brain (e4b)
to quickly score each chunk for usefulness before storing.
"""

from __future__ import annotations

import re
from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="knowledge.quality")

_SCORE_PROMPT = (
    "Rate the following text for quality and usefulness as training data "
    "for a language model. Consider:\n"
    "- Information density (contains real facts/instructions/code)\n"
    "- Clarity (well-structured, coherent)\n"
    "- Completeness (self-contained, not a fragment)\n"
    "- Noise level (minimal boilerplate, ads, navigation text)\n\n"
    "Score from 1 to 5:\n"
    "1 = Useless (boilerplate, ads, navigation only)\n"
    "2 = Low quality (mostly noise, few useful bits)\n"
    "3 = Acceptable (some useful info, some noise)\n"
    "4 = Good (mostly useful, well-structured)\n"
    "5 = Excellent (high-quality, dense information)\n\n"
    "Reply with ONLY a single integer (1-5), nothing else.\n\n"
    "Text:\n{chunk}"
)

_RE_DIGIT = re.compile(r"[1-5]")


class QualityScorer:
    """LLM rates content usefulness 1-5."""

    async def score(self, chunk: str, api: Any) -> int:
        """Ask LLM to rate *chunk* 1-5, parse integer response.

        Returns 3 (neutral) on error so content is neither eagerly
        accepted nor rejected when the LLM is unavailable.
        """
        prompt = _SCORE_PROMPT.format(chunk=chunk[:1500])

        try:
            response = await api.llm_chat(
                messages=[{"role": "user", "content": prompt}],
                brain="e4b",
                max_tokens=5,
                temperature=0.0,
            )
        except Exception as exc:
            log.error("quality.llm_error", error=str(exc))
            return 3

        # Ersten Digit 1-5 aus der Antwort extrahieren
        cleaned = response.strip()
        match = _RE_DIGIT.search(cleaned)
        if match:
            score_val = int(match.group(0))
            log.debug("quality.scored", score=score_val, raw=cleaned)
            return score_val

        log.warning("quality.parse_failed", raw=cleaned, fallback=3)
        return 3
