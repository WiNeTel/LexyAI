"""
Lexy AI - Web Crawler Plugin.

Provides two LLM-callable tools:

* ``web_search`` – search the web via a local SearXNG instance.
* ``web_fetch``  – fetch a URL and extract its plain-text content.

No external APIs or keys required – everything runs against the local
SearXNG at ``http://127.0.0.1:7899`` by default.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="web_crawler")

# ─── Tool schemas ────────────────────────────────────────────────────────

WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query",
        },
        "max_results": {
            "type": "integer",
            "description": "Max results (default 5)",
        },
    },
    "required": ["query"],
}

WEB_FETCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "URL to fetch",
        },
    },
    "required": ["url"],
}

# ─── HTML → plain-text helpers ───────────────────────────────────────────

# Regex patterns compiled once at module level
_RE_SCRIPT_STYLE = re.compile(
    r"<\s*(script|style|noscript)[^>]*>.*?</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MULTI_WHITESPACE = re.compile(r"[ \t]+")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
_RE_HTML_ENTITIES: dict[str, str] = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
}


def _strip_html(raw: str) -> str:
    """Regex-based HTML → plain-text conversion (no extra dependencies)."""
    # Entferne script/style/noscript Blöcke komplett
    text = _RE_SCRIPT_STYLE.sub("", raw)
    # Zeilenumbrüche bei Block-Elementen einfügen
    text = re.sub(r"<\s*/?\s*(br|p|div|li|tr|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
    # Alle HTML-Tags entfernen
    text = _RE_HTML_TAG.sub("", text)
    # HTML-Entities dekodieren
    for entity, char in _RE_HTML_ENTITIES.items():
        text = text.replace(entity, char)
    # Numerische Entities dekodieren (&#123; und &#x7b;)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    # Whitespace normalisieren
    text = _RE_MULTI_WHITESPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()


def _extract_title(raw: str) -> str:
    """Pull the <title> from raw HTML, or return an empty string."""
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    if match:
        title = _RE_HTML_TAG.sub("", match.group(1))
        for entity, char in _RE_HTML_ENTITIES.items():
            title = title.replace(entity, char)
        return title.strip()
    return ""


# ─── Plugin ──────────────────────────────────────────────────────────────


class WebCrawlerPlugin(BasePlugin):
    """Web search + page fetching via local SearXNG."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._client: httpx.AsyncClient | None = None
        self._searxng_url: str = "http://127.0.0.1:7899"
        self._max_results: int = 5
        self._fetch_timeout: float = 15.0
        self._max_content_length: int = 4000

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._searxng_url = str(config.get("searxng_url", "http://127.0.0.1:7899")).rstrip("/")
        self._max_results = int(config.get("max_results", 5))
        self._fetch_timeout = float(config.get("fetch_timeout", 15.0))
        self._max_content_length = int(config.get("max_content_length", 4000))

        self._client = httpx.AsyncClient(
            timeout=self._fetch_timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "LexyAI/1.0 (local web crawler)",
                "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9",
            },
        )
        log.info(
            "web_crawler.loaded",
            searxng_url=self._searxng_url,
            max_results=self._max_results,
            fetch_timeout=self._fetch_timeout,
            max_content_length=self._max_content_length,
        )

    async def on_enable(self) -> None:
        self.api.register_tool(
            name="web_search",
            handler=self._handle_search,
            description=(
                "Search the web using SearXNG. Returns a list of results with "
                "title, URL and snippet. Use this to find information online."
            ),
            schema=WEB_SEARCH_SCHEMA,
        )
        self.api.register_tool(
            name="web_fetch",
            handler=self._handle_fetch,
            description=(
                "Fetch a web page and extract its plain-text content. "
                "Use this to read the full content of a URL found via web_search."
            ),
            schema=WEB_FETCH_SCHEMA,
        )
        log.info("web_crawler.enabled", tools=["web_search", "web_fetch"])

    async def on_disable(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("web_crawler.disabled")

    # ─── Tool: web_search ────────────────────────────────────────────

    async def _handle_search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Search via SearXNG and return structured results."""
        if self._client is None:
            return {"error": "Web crawler not loaded"}

        limit = max_results if max_results is not None else self._max_results
        limit = max(1, min(20, limit))

        log.info("web_search.start", query=query, max_results=limit)

        try:
            resp = await self._client.get(
                f"{self._searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": "de",
                },
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            log.error("web_search.timeout", query=query)
            return {"error": f"SearXNG request timed out for query: {query}"}
        except httpx.HTTPError as exc:
            log.error("web_search.http_error", query=query, error=str(exc))
            return {"error": f"SearXNG request failed: {exc}"}

        try:
            data = resp.json()
        except ValueError:
            log.error("web_search.invalid_json", query=query)
            return {"error": "SearXNG returned invalid JSON"}

        raw_results: list[dict[str, Any]] = data.get("results", [])
        results: list[dict[str, str]] = []
        for item in raw_results[:limit]:
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", "")),
                }
            )

        log.info(
            "web_search.done",
            query=query,
            result_count=len(results),
            total_available=len(raw_results),
        )
        return {"query": query, "results": results}

    # ─── Tool: web_fetch ─────────────────────────────────────────────

    async def _handle_fetch(self, url: str) -> dict[str, Any]:
        """Fetch a URL and return extracted plain-text content."""
        if self._client is None:
            return {"error": "Web crawler not loaded"}

        log.info("web_fetch.start", url=url)

        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.TimeoutException:
            log.error("web_fetch.timeout", url=url)
            return {"error": f"Request timed out: {url}"}
        except httpx.HTTPError as exc:
            log.error("web_fetch.http_error", url=url, error=str(exc))
            return {"error": f"Fetch failed: {exc}"}

        content_type = resp.headers.get("content-type", "")
        raw_body = resp.text

        # HTML-Seite: Titel extrahieren und Tags strippen
        if "html" in content_type.lower():
            title = _extract_title(raw_body)
            text = _strip_html(raw_body)
        else:
            # Plain text / JSON / sonstiges: direkt verwenden
            title = ""
            text = raw_body

        # Auf max_content_length kürzen (am Wortende abschneiden)
        if len(text) > self._max_content_length:
            truncated = text[: self._max_content_length]
            # Am letzten Leerzeichen abschneiden, damit kein Wort zerhackt wird
            last_space = truncated.rfind(" ")
            if last_space > self._max_content_length * 0.8:
                truncated = truncated[:last_space]
            text = truncated + "\n\n[... truncated]"

        log.info(
            "web_fetch.done",
            url=url,
            title=title,
            content_length=len(text),
        )

        return {
            "url": url,
            "title": title,
            "content": text,
        }
