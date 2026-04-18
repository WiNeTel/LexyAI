"""
Lexy AI - Dashboard Memory Stats Widget.

Reports item counts for every ChromaDB collection and the FTS5 mirror.
Uses the MemoryManager's internal ``_collections`` dict to call ``.count()``
on each ChromaDB ``Collection`` object, plus a row count on the FTS table.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.memory_stats")


class MemoryStatsWidget(BaseWidget):
    """ChromaDB collection counts + FTS5 row count."""

    widget_id: str = "memory_stats"
    title: str = "Memory"
    default_size: tuple[int, int] = (2, 2)
    refresh_interval: float = 60.0

    def __init__(self, api: Any) -> None:
        super().__init__(api)

    async def get_data(self) -> dict[str, Any]:
        """Count items in each ChromaDB collection and the FTS5 mirror."""
        memory = self._api._app.memory
        if memory is None:
            log.debug("widget.memory_stats.no_memory")
            return {
                "available": False,
                "collections": {},
                "total": 0,
                "fts_count": 0,
            }

        # --- ChromaDB collection counts ---
        counts: dict[str, int] = {}
        total: int = 0
        # Snapshot the name list so we can refresh entries in-place without
        # mutating during iteration.
        names = list(memory._collections.keys())
        for name in names:
            collection = memory._collections.get(name)
            if collection is None:
                counts[name] = 0
                continue
            try:
                count = collection.count()
            except Exception as exc:  # noqa: BLE001
                # Stale collection handle (server-side wipe etc.). Try to
                # refresh once — if that succeeds, re-count; otherwise
                # report 0 for this widget refresh cycle.
                refreshed = None
                if hasattr(memory, "_refresh_collection"):
                    try:
                        refreshed = memory._refresh_collection(name)
                    except Exception:  # noqa: BLE001
                        refreshed = None
                if refreshed is not None:
                    try:
                        count = refreshed.count()
                    except Exception as exc2:  # noqa: BLE001
                        log.warning(
                            "widget.memory_stats.count_failed_after_refresh",
                            collection=name,
                            error=str(exc2),
                        )
                        count = 0
                else:
                    log.warning(
                        "widget.memory_stats.count_failed",
                        collection=name,
                        error=str(exc),
                    )
                    count = 0
            counts[name] = int(count)
            total += int(count)

        # --- FTS5 row count ---
        fts_count: int = 0
        if memory._fts is not None:
            try:
                cursor = await memory._fts.execute(
                    "SELECT COUNT(*) FROM items_fts"
                )
                row = await cursor.fetchone()
                await cursor.close()
                fts_count = int(row[0]) if row else 0
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "widget.memory_stats.fts_count_failed",
                    error=str(exc),
                )

        return {
            "available": True,
            "collections": counts,
            "total": total,
            "fts_count": fts_count,
        }
