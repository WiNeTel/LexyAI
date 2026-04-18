"""
Lexy AI - Knowledge Acquisition: Crawler.

Orchestrates search and fetch using the web_crawler plugin.
Delegates all HTTP work to web_crawler so we inherit its rate
limiting, headers, and HTML stripping.
"""

from __future__ import annotations

from typing import Any

from lexy_core.utils.logging import get_logger

log = get_logger(module="knowledge.crawler")


class KnowledgeCrawler:
    """Orchestrates search and fetch using the web_crawler plugin."""

    def __init__(self, web_crawler_plugin: Any) -> None:
        self._crawler = web_crawler_plugin

    async def search_topic(
        self, topic: str, max_pages: int = 10
    ) -> list[dict[str, str]]:
        """Search SearXNG for *topic*, return list of {url, title, snippet}.

        Delegates to ``web_crawler._handle_search``. Filters out results
        without a URL and deduplicates by URL.
        """
        log.info("knowledge.search_topic", topic=topic, max_pages=max_pages)

        result = await self._crawler._handle_search(
            query=topic, max_results=max_pages
        )

        if "error" in result:
            log.error("knowledge.search_error", error=result["error"])
            return []

        raw_results: list[dict[str, Any]] = result.get("results", [])
        seen_urls: set[str] = set()
        deduplicated: list[dict[str, str]] = []

        for item in raw_results:
            url = item.get("url", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduplicated.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                }
            )

        log.info(
            "knowledge.search_done",
            topic=topic,
            results_found=len(deduplicated),
        )
        return deduplicated

    async def fetch_page(self, url: str) -> dict[str, str] | None:
        """Fetch and clean a page via web_crawler.

        Returns ``{url, title, content}`` or ``None`` on error.
        """
        log.info("knowledge.fetch_page", url=url)

        result = await self._crawler._handle_fetch(url=url)

        if "error" in result:
            log.warning("knowledge.fetch_error", url=url, error=result["error"])
            return None

        content = result.get("content", "").strip()
        if not content:
            log.warning("knowledge.fetch_empty", url=url)
            return None

        log.info(
            "knowledge.fetch_done",
            url=url,
            content_length=len(content),
        )
        return {
            "url": result.get("url", url),
            "title": result.get("title", ""),
            "content": content,
        }
