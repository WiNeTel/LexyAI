"""
Per-sub-query researcher.

For each sub-query the pipeline:

1. Calls ``web_search`` (via the web_crawler plugin's tool registry)
   to get candidate URLs.
2. Fetches the top N pages in parallel with ``web_fetch`` (also from
   web_crawler).
3. Asks a small LLM (default ``e4b``) to extract relevant quotes per
   page and score the page's relevance to the sub-query.
4. Records everything in a :class:`SubqueryResult`.

The class is decoupled from the plugin: it takes a generic
``call_tool`` callable + ``llm_chat`` callable so unit tests can inject
fakes. The plugin wires those to ``api.call_tool`` and ``api.llm_chat``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable

from .state import SourceHit, SubqueryResult


log = logging.getLogger(__name__)


LLMChat = Callable[..., Awaitable[str]]
ToolRunner = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


# Drop pages whose extractor relevance is under this threshold from the
# synthesis input. Keeps low-signal noise out of the final report.
_MIN_RELEVANCE_FOR_SYNTHESIS: float = 0.25


class Researcher:
    """Runs one sub-query end-to-end."""

    _EXTRACTOR_SYSTEM = (
        "Du bist ein Recherche-Extractor. Eine Web-Seite wurde zu einer "
        "konkreten Sub-Frage gefetcht. Deine Aufgabe:\n"
        "1) Bewerte die Relevanz der Seite für die Frage als Zahl 0..1 "
        "(0 = irrelevant, 1 = perfekter Treffer).\n"
        "2) Picke maximal {max_quotes} faktentreue Kern-Aussagen "
        "(Originaltext oder eng paraphrasiert), die direkt zur Frage "
        "beitragen. Keine Spekulation, keine Marketing-Floskeln.\n\n"
        "Antworte AUSSCHLIESSLICH als JSON:\n"
        '{{"relevance": 0.0..1.0, "quotes": ["...", "..."]}}\n'
        "KEIN Markdown, KEIN Code-Fence."
    )

    def __init__(
        self,
        *,
        call_tool: ToolRunner,
        llm_chat: LLMChat,
        search_results_per_query: int = 8,
        fetch_per_query: int = 4,
        fetch_concurrency: int = 2,
        page_text_max_chars: int = 6000,
        extractor_brain: str = "e4b",
        extractor_max_tokens: int = 400,
        extractor_temperature: float = 0.2,
        max_quotes_per_page: int = 4,
    ) -> None:
        self._call_tool = call_tool
        self._chat = llm_chat
        self._search_results = max(1, int(search_results_per_query))
        self._fetch_n = max(1, int(fetch_per_query))
        self._fetch_concurrency = max(1, int(fetch_concurrency))
        self._page_max = max(500, int(page_text_max_chars))
        self._brain = extractor_brain
        self._extract_max_tokens = max(120, int(extractor_max_tokens))
        self._extract_temp = float(extractor_temperature)
        self._max_quotes = max(1, int(max_quotes_per_page))

    async def run(
        self,
        *,
        query: str,
        topic: str = "",
        language: str = "de",
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> SubqueryResult:
        """Run the search→fetch→extract chain for one sub-query."""
        result = SubqueryResult(query=query)

        # ── 1) Web search ────────────────────────────────────────
        try:
            search = await self._call_tool(
                "web_search",
                {"query": query, "max_results": self._search_results},
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"search_failed: {exc}"
            return result
        if not search.get("ok") or not isinstance(search.get("data"), dict):
            result.error = (
                f"search_failed: {search.get('error') or 'no data'}"
            )
            return result
        hits = search["data"].get("results", [])
        if not isinstance(hits, list) or not hits:
            result.error = "no_search_results"
            return result

        # Take only the top N for fetching, but keep all in the result
        # for context (the synthesiser may want to know what was on offer).
        for h in hits[: self._fetch_n]:
            url = str((h or {}).get("url") or "").strip()
            if not url:
                continue
            result.sources.append(
                SourceHit(
                    url=url,
                    title=str((h or {}).get("title") or "").strip(),
                    subquery=query,
                    snippet=str((h or {}).get("snippet") or "")[:300],
                )
            )

        if on_event is not None:
            await on_event(
                {"phase": "search_done", "query": query, "candidates": len(result.sources)}
            )

        # ── 2) Fetch + extract per source (concurrency-limited) ──
        sem = asyncio.Semaphore(self._fetch_concurrency)

        async def _process(source: SourceHit) -> None:
            async with sem:
                await self._fetch_and_extract(
                    source=source, topic=topic, query=query,
                )
                if on_event is not None:
                    await on_event(
                        {
                            "phase": "source_processed",
                            "query": query,
                            "url": source.url,
                            "fetched": source.fetched,
                            "relevance": source.relevance,
                            "quotes": len(source.extracted),
                        }
                    )

        await asyncio.gather(
            *(_process(s) for s in result.sources),
            return_exceptions=True,
        )
        result.completed = True
        return result

    # ─── Internals ──────────────────────────────────────────────────

    async def _fetch_and_extract(
        self,
        *,
        source: SourceHit,
        topic: str,
        query: str,
    ) -> None:
        try:
            fetched = await self._call_tool("web_fetch", {"url": source.url})
        except Exception as exc:  # noqa: BLE001
            source.fetch_error = f"fetch_exception: {exc}"
            return
        if not fetched.get("ok") or not isinstance(fetched.get("data"), dict):
            source.fetch_error = (
                f"fetch_failed: {fetched.get('error') or 'no data'}"
            )
            return
        page = fetched["data"]
        text = str(page.get("content") or "").strip()
        if not source.title:
            source.title = str(page.get("title") or "").strip()
        if not text:
            source.fetched = False
            source.fetch_error = "empty_page_content"
            return
        source.chars = len(text)
        source.fetched = True

        # Truncate before sending to the extractor.
        if len(text) > self._page_max:
            text = text[: self._page_max].rstrip() + "\n[...truncated]"

        # ── Extractor LLM call ──
        system = self._EXTRACTOR_SYSTEM.format(max_quotes=self._max_quotes)
        user = (
            f"Sub-Frage: {query}\n"
            f"Übergeordnetes Thema: {topic or query}\n"
            f"URL: {source.url}\n"
            f"Titel: {source.title or '(unknown)'}\n\n"
            f"## Seiteninhalt\n{text}"
        )
        try:
            raw = await self._chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                brain=self._brain,
                max_tokens=self._extract_max_tokens,
                temperature=self._extract_temp,
                thinking=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "deep_research.extract_failed url=%s err=%s",
                source.url, exc,
            )
            return
        relevance, quotes = parse_extract(raw)
        source.relevance = max(0.0, min(1.0, relevance))
        source.extracted = quotes[: self._max_quotes]


# ─── Parser ─────────────────────────────────────────────────────────


def parse_extract(raw: str) -> tuple[float, list[str]]:
    """Best-effort parser for ``{relevance: float, quotes: [str]}`` blobs."""
    text = (raw or "").strip()
    if not text:
        return 0.0, []
    # Strip code fence.
    if text.startswith("```"):
        first_break = text.find("\n")
        if first_break != -1:
            text = text[first_break + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try strict.
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: find the first { ... } substring.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                parsed = None

    if not isinstance(parsed, dict):
        return 0.0, []

    rel_raw = parsed.get("relevance", 0.0)
    try:
        relevance = float(rel_raw)
    except (TypeError, ValueError):
        relevance = 0.0

    quotes_raw = parsed.get("quotes") or parsed.get("snippets") or []
    if not isinstance(quotes_raw, list):
        quotes_raw = []
    quotes: list[str] = []
    for q in quotes_raw:
        q_str = str(q or "").strip()
        if q_str:
            # Long quotes get truncated. The synthesiser doesn't need
            # paragraph-long blocks per source.
            if len(q_str) > 600:
                q_str = q_str[:600].rstrip() + "…"
            quotes.append(q_str)
    return relevance, quotes


def relevant_sources(result: SubqueryResult) -> list[SourceHit]:
    """Filter sources for synthesis input — drop the obvious noise."""
    return [
        s for s in result.sources
        if s.fetched and s.extracted and s.relevance >= _MIN_RELEVANCE_FOR_SYNTHESIS
    ]


__all__ = ["Researcher", "parse_extract", "relevant_sources"]
