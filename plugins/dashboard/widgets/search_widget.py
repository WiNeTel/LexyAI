"""
Lexy AI - Dashboard Search Widget.

Provides a quick-search interface across all memory collections.  By
default ``get_data()`` returns an empty result set; actual results are
populated when the user triggers a search via ``search(query)``.

The DashboardPlugin wires ``search()`` as a dedicated WS handler so the
frontend can request results on demand.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

from .base_widget import BaseWidget

log = get_logger(module="widget.search")

_MAX_RESULTS: int = 20


class SearchWidget(BaseWidget):
    """On-demand memory search."""

    widget_id: str = "search"
    title: str = "Suche"
    default_size: tuple[int, int] = (3, 2)
    refresh_interval: float = 0.0  # on-demand only

    def __init__(self, api: Any) -> None:
        super().__init__(api)
        self._last_query: str = ""
        self._last_results: list[dict[str, Any]] = []

    # ─── Default data (empty) ───────────────────────────────────────

    async def get_data(self) -> dict[str, Any]:
        """Return the most recent search results (or empty on first load)."""
        return {
            "results": self._last_results,
            "query": self._last_query,
            "count": len(self._last_results),
        }

    # ─── Active search ──────────────────────────────────────────────

    async def search(self, query: str) -> dict[str, Any]:
        """
        Run a hybrid search across all memory collections.

        Called by the DashboardPlugin's WS handler when the user types
        a search query in the dashboard.
        """
        query = query.strip()
        if not query:
            self._last_query = ""
            self._last_results = []
            return await self.get_data()

        log.debug("widget.search.query", query=query)

        results: list[dict[str, Any]] = []

        # Hybrid recall (vector + BM25) across all collections
        try:
            recalled = await self._api.memory_recall(
                query=query, collection=None, limit=_MAX_RESULTS
            )
            for item in recalled:
                results.append({
                    "id": item.get("id", ""),
                    "collection": item.get("collection", ""),
                    "content": str(item.get("content", ""))[:300],
                    "score": round(float(item.get("score", 0.0)), 3),
                    "metadata": item.get("metadata", {}),
                })
        except Exception as exc:  # noqa: BLE001
            log.warning("widget.search.recall_failed", error=str(exc))

        # Supplement with pure FTS if hybrid returned few hits
        if len(results) < 5:
            try:
                fts_hits = await self._api.memory_search_fts(
                    query=query, limit=_MAX_RESULTS - len(results)
                )
                seen_ids = {r["id"] for r in results}
                for item in fts_hits:
                    item_id = item.get("id", "")
                    if item_id in seen_ids:
                        continue
                    results.append({
                        "id": item_id,
                        "collection": item.get("collection", ""),
                        "content": str(item.get("content", ""))[:300],
                        "score": round(float(item.get("bm25_score", 0.0)), 3),
                        "metadata": {},
                    })
                    seen_ids.add(item_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("widget.search.fts_failed", error=str(exc))

        self._last_query = query
        self._last_results = results[:_MAX_RESULTS]

        return await self.get_data()
