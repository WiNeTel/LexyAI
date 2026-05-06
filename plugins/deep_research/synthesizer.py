"""
Synthesiser — turns the extracted snippets into a final markdown report.

Strategy:

* Collect every relevant ``SourceHit`` (relevance ≥ threshold, has
  extracted quotes), de-duplicated by URL.
* Number them ``[1]``, ``[2]``, … so the LLM can cite them in-line.
* Hand the LLM a tight system prompt: "write a structured German
  report with section per sub-query, cite using [N]".
* Append a Sources section with one numbered link per source.

The synthesiser is the *only* component that produces user-facing prose
in this pipeline — everything else is internal plumbing. So this is
where we put real effort into making the output feel like a research
note, not like raw search dumps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .researcher import relevant_sources
from .state import ResearchTask, SourceHit, SubqueryResult


log = logging.getLogger(__name__)


LLMChat = Callable[..., Awaitable[str]]


@dataclass
class _NumberedSource:
    n: int
    url: str
    title: str
    quotes: list[str] = field(default_factory=list)


class Synthesizer:
    """Builds the final markdown report."""

    _SYSTEM_PROMPT = (
        "Du bist Lexys Recherche-Synthese. Du bekommst eine User-Frage, "
        "eine Liste von Sub-Queries und pro Quelle ein paar Zitat-"
        "Schnipsel mit Quellennummer.\n\n"
        "Deine Aufgabe: Schreibe einen lesbaren, strukturierten Bericht "
        "in {language}.\n\n"
        "Format:\n"
        "1) Eine 2-3 Sätze lange **Zusammenfassung ganz oben** (TL;DR).\n"
        "2) Anschliessend eine Sektion pro Sub-Query mit der Sub-Query "
        "als ## Heading. In den Sektionen schreibst du in eigenen "
        "Worten, was die Quellen sagen — flüssig, nicht als "
        "Stichpunkt-Liste.\n"
        "3) Zitiere am Satz-Ende mit der Quellennummer in eckigen "
        "Klammern wie ``[1]`` oder ``[1, 3]``. Erfinde KEINE "
        "Quellennummern — nur die, die unten in der Quellen-Liste "
        "stehen.\n"
        "4) Am Ende: ## Quellen, mit nummerierter Liste in der Form "
        "``[N] Titel — URL``.\n"
        "5) Wenn etwas widersprüchlich ist, sag das offen statt es "
        "wegzubügeln. Wenn etwas in den Quellen nicht steht, behaupte "
        "es nicht.\n"
        "6) Keine Halluzination. Nur Aussagen, die durch die "
        "übergebenen Schnipsel gedeckt sind."
    )

    def __init__(
        self,
        *,
        llm_chat: LLMChat,
        brain: str = "a4b",
        max_tokens: int = 2400,
        temperature: float = 0.5,
    ) -> None:
        self._chat = llm_chat
        self._brain = brain
        self._max_tokens = max(400, int(max_tokens))
        self._temperature = float(temperature)

    async def synthesise(
        self,
        task: ResearchTask,
    ) -> str:
        """Return the final markdown report. Empty string on full failure."""
        numbered = _number_sources(task.subquery_results)
        if not numbered:
            return _no_sources_fallback(task)

        sources_block = _format_sources_for_prompt(numbered)
        subqueries_block = _format_subqueries_for_prompt(
            task.subquery_results, numbered,
        )

        system = self._SYSTEM_PROMPT.format(language=_lang_label(task.language))
        user = (
            f"## Original-Frage\n{task.topic.strip()}\n\n"
            f"## Sub-Queries\n{subqueries_block}\n\n"
            f"## Quellen\n{sources_block}\n\n"
            "Schreibe jetzt den Bericht."
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
            log.warning("deep_research.synth_failed err=%s", exc)
            # Fall back to a deterministic layout so the user still gets
            # _something_ usable when the LLM call breaks.
            return _deterministic_fallback(task, numbered)

        report = (raw or "").strip()
        if not report:
            return _deterministic_fallback(task, numbered)

        # Always make sure the sources list is present and complete —
        # the LLM occasionally truncates it when max_tokens runs short.
        if "## Quellen" not in report:
            report += "\n\n" + _format_sources_appendix(numbered)
        return report


# ─── Helpers ────────────────────────────────────────────────────────


def _number_sources(
    subquery_results: list[SubqueryResult],
) -> list[_NumberedSource]:
    """Flatten + de-duplicate sources by URL, assign 1-based numbers."""
    seen: dict[str, _NumberedSource] = {}
    counter = 1
    for sr in subquery_results:
        for source in relevant_sources(sr):
            existing = seen.get(source.url)
            if existing is not None:
                # Merge quotes from both occurrences.
                for q in source.extracted:
                    if q not in existing.quotes:
                        existing.quotes.append(q)
                continue
            seen[source.url] = _NumberedSource(
                n=counter,
                url=source.url,
                title=source.title or source.url,
                quotes=list(source.extracted),
            )
            counter += 1
    return list(seen.values())


def _format_sources_for_prompt(numbered: list[_NumberedSource]) -> str:
    lines: list[str] = []
    for s in numbered:
        lines.append(f"[{s.n}] {s.title} — {s.url}")
        for q in s.quotes:
            lines.append(f"    » {q}")
    return "\n".join(lines)


def _format_subqueries_for_prompt(
    subquery_results: list[SubqueryResult],
    numbered: list[_NumberedSource],
) -> str:
    url_to_n = {s.url: s.n for s in numbered}
    lines: list[str] = []
    for sr in subquery_results:
        used = sorted(
            {
                url_to_n[s.url]
                for s in relevant_sources(sr)
                if s.url in url_to_n
            }
        )
        used_str = (
            ", ".join(f"[{n}]" for n in used) if used else "(keine relevante Quelle)"
        )
        lines.append(f"- {sr.query} → {used_str}")
    return "\n".join(lines)


def _format_sources_appendix(numbered: list[_NumberedSource]) -> str:
    lines = ["## Quellen"]
    for s in numbered:
        title = s.title or s.url
        lines.append(f"[{s.n}] {title} — {s.url}")
    return "\n".join(lines)


def _no_sources_fallback(task: ResearchTask) -> str:
    """No relevant pages survived extraction — give the user that fact
    plus what was attempted, so they can refine the topic."""
    parts = [
        f"# Recherche: {task.topic}\n",
        "Ich habe für deine Frage keine relevanten Quellen gefunden.",
        "Was ich gesucht habe:",
    ]
    for q in task.plan or [task.topic]:
        parts.append(f"- {q}")
    parts.append(
        "\nVielleicht hilft eine konkretere Formulierung der Frage, "
        "ein anderer Sprach-Filter, oder mehr Sub-Queries."
    )
    return "\n".join(parts)


def _deterministic_fallback(
    task: ResearchTask, numbered: list[_NumberedSource]
) -> str:
    """Used when the synth LLM call breaks. Plain layout, no prose."""
    parts = [
        f"# Recherche: {task.topic}",
        "",
        "_(LLM-Synthese fehlgeschlagen — hier sind die rohen Funde.)_",
        "",
    ]
    for sr in task.subquery_results:
        parts.append(f"## {sr.query}")
        url_to_n = {s.url: s.n for s in numbered}
        rels = relevant_sources(sr)
        if not rels:
            parts.append("(keine relevanten Quellen)\n")
            continue
        for source in rels:
            n = url_to_n.get(source.url)
            label = f"[{n}]" if n else ""
            parts.append(f"**{label} {source.title or source.url}**")
            for q in source.extracted:
                parts.append(f"> {q}")
            parts.append("")
    parts.append(_format_sources_appendix(numbered))
    return "\n".join(parts)


def _lang_label(language: str) -> str:
    lang = (language or "de").lower()
    return {
        "de": "Deutsch",
        "en": "Englisch",
    }.get(lang, language)


__all__ = ["Synthesizer"]
