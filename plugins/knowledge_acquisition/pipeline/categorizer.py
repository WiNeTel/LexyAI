"""
Lexy AI - Knowledge Acquisition: Content Categorizer.

LLM-based category assignment.  Sends each chunk to the fast brain (e4b)
with a classification prompt and parses the single-word response.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="knowledge.categorizer")

_CLASSIFY_PROMPT = (
    "You are a strict text classifier. "
    "Classify the following text into EXACTLY ONE of these categories: {categories}.\n\n"
    "Rules:\n"
    "- Reply with ONLY the category name, nothing else.\n"
    "- The category MUST be one from the list above.\n"
    "- If unsure, use 'general'.\n\n"
    "Text:\n{chunk}"
)


class ContentCategorizer:
    """LLM-based category assignment."""

    async def categorize(
        self,
        chunk: str,
        categories: list[str],
        api: Any,
    ) -> str:
        """Send *chunk* to LLM (e4b) and get a category label back.

        Falls back to ``"general"`` when the LLM response does not match
        any of the provided categories.
        """
        if not categories:
            return "general"

        categories_str = ", ".join(categories)
        prompt = _CLASSIFY_PROMPT.format(
            categories=categories_str,
            chunk=chunk[:1500],  # Chunk kuerzen um Tokens zu sparen
        )

        try:
            response = await api.llm_chat(
                messages=[{"role": "user", "content": prompt}],
                brain="e4b",
                max_tokens=20,
                temperature=0.0,
            )
        except Exception as exc:
            log.error("categorizer.llm_error", error=str(exc))
            return "general"

        # Antwort bereinigen und gegen Kategorien pruefen
        label = response.strip().lower().replace('"', "").replace("'", "")
        # Manchmal antwortet das LLM mit "Category: python_docs"
        if ":" in label:
            label = label.split(":", 1)[1].strip()

        # Exakten Match suchen
        categories_lower = {c.lower(): c for c in categories}
        if label in categories_lower:
            return categories_lower[label]

        # Teilmatch: Label ist in einer Kategorie enthalten oder umgekehrt
        for cat_lower, cat_original in categories_lower.items():
            if label in cat_lower or cat_lower in label:
                log.debug(
                    "categorizer.partial_match",
                    raw_label=label,
                    matched=cat_original,
                )
                return cat_original

        log.warning(
            "categorizer.unknown_label",
            raw_label=label,
            fallback="general",
        )
        return "general"
