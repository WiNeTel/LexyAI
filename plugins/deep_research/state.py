"""
Deep-Research task state.

The plugin keeps a small dataclass per active research task in memory.
Every WS-broadcast about progress is computed from the current state of
this object — no separate "progress dict" passed around. That way
reconnecting clients can ask ``deep_research_status({research_id})``
and get a coherent snapshot at any point in the lifecycle.

States
------
- ``pending``       — accepted, before the planner ran
- ``planning``      — LLM is decomposing the topic
- ``searching``     — sub-queries running in parallel
- ``synthesising``  — final report being assembled
- ``done``          — finished successfully
- ``failed``        — unrecoverable error
- ``cancelled``     — user invoked ``deep_research_stop``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


ResearchState = Literal[
    "pending", "planning", "searching", "synthesising",
    "done", "failed", "cancelled",
]


@dataclass
class SourceHit:
    """One web source visited during the search phase."""

    url: str
    title: str = ""
    subquery: str = ""              # which sub-query produced this hit
    snippet: str = ""               # search-result snippet (short)
    fetched: bool = False           # did we actually fetch the page?
    fetch_error: str = ""           # populated when fetched is False
    extracted: list[str] = field(default_factory=list)  # relevant quotes
    relevance: float = 0.0          # 0..1 from extractor; 0 = ignored
    chars: int = 0                  # length of fetched text

    def to_public(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "subquery": self.subquery,
            "snippet": self.snippet,
            "fetched": self.fetched,
            "fetch_error": self.fetch_error,
            "extracted": list(self.extracted),
            "relevance": round(float(self.relevance), 3),
            "chars": self.chars,
        }


@dataclass
class SubqueryResult:
    """Results for one sub-query."""

    query: str
    sources: list[SourceHit] = field(default_factory=list)
    completed: bool = False
    error: str = ""

    def to_public(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sources": [s.to_public() for s in self.sources],
            "source_count": len(self.sources),
            "completed": self.completed,
            "error": self.error,
        }


@dataclass
class ResearchTask:
    """One in-flight deep-research request."""

    research_id: str
    topic: str
    language: str = "de"
    state: ResearchState = "pending"
    plan: list[str] = field(default_factory=list)
    subquery_results: list[SubqueryResult] = field(default_factory=list)
    report: str = ""                  # final markdown synthesis
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    last_error: str = ""
    session_id: str = ""

    def progress(self) -> dict[str, Any]:
        completed = sum(1 for r in self.subquery_results if r.completed)
        sources_total = sum(len(r.sources) for r in self.subquery_results)
        sources_fetched = sum(
            1 for r in self.subquery_results
            for s in r.sources if s.fetched
        )
        sources_relevant = sum(
            1 for r in self.subquery_results
            for s in r.sources if s.extracted
        )
        return {
            "subqueries_completed": completed,
            "subqueries_total": len(self.subquery_results) or len(self.plan),
            "sources_total": sources_total,
            "sources_fetched": sources_fetched,
            "sources_relevant": sources_relevant,
        }

    def to_public(self) -> dict[str, Any]:
        return {
            "research_id": self.research_id,
            "topic": self.topic,
            "language": self.language,
            "state": self.state,
            "plan": list(self.plan),
            "subqueries": [r.to_public() for r in self.subquery_results],
            "progress": self.progress(),
            "report": self.report,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "session_id": self.session_id,
        }
