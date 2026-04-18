"""
Lexy AI - Knowledge Acquisition: Content Deduplicator.

Checks for duplicate content using embedding similarity via ChromaDB.
Queries the ``knowledge`` collection and flags content as duplicate when
its cosine similarity exceeds the configured threshold.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="knowledge.dedup")


class ContentDeduplicator:
    """Check for duplicate content using embedding similarity."""

    async def is_duplicate(
        self,
        text: str,
        threshold: float,
        api: Any,
    ) -> bool:
        """Query ChromaDB 'knowledge' collection for similar content.

        Returns ``True`` when the most similar existing entry has a
        similarity score above *threshold* (0.0 .. 1.0).

        If the collection is empty or memory is unavailable, returns ``False``
        so that new content can always be ingested into an empty system.
        """
        try:
            results = await api.memory_recall(
                query=text[:500],  # Nur Anfang fuer Embedding
                collection="knowledge",
                limit=1,
            )
        except Exception as exc:
            log.warning("dedup.recall_error", error=str(exc))
            return False

        if not results:
            return False

        # ChromaDB recall liefert Items mit 'score' Feld (0..1, hoeher = aehnlicher)
        best = results[0]
        score: float = float(best.get("score", 0.0))

        is_dup = score >= threshold
        if is_dup:
            log.info(
                "dedup.duplicate_found",
                score=round(score, 3),
                threshold=threshold,
                existing_id=best.get("id", "?"),
            )
        else:
            log.debug(
                "dedup.no_duplicate",
                score=round(score, 3),
                threshold=threshold,
            )
        return is_dup
