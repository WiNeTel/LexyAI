"""
Tests for the deep_research plugin.

Four classes — one per pipeline stage plus a state-roundtrip class:

* :class:`TestPlanParser` / :class:`TestPlanner` — sub-query
  decomposition, tolerant parser.
* :class:`TestExtractParser` / :class:`TestResearcher` — search +
  fetch + extract pipeline with deterministic fakes for the
  ``call_tool`` and ``llm_chat`` callables.
* :class:`TestSynthesizer` — markdown layout, citation correctness,
  fallback when the LLM call fails.
* :class:`TestStateModel` — :class:`ResearchTask` snapshots survive
  a ``to_public()`` round-trip and the progress counters work.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from plugins.deep_research.planner import Planner, parse_subqueries
from plugins.deep_research.researcher import (
    Researcher,
    parse_extract,
    relevant_sources,
)
from plugins.deep_research.state import (
    ResearchTask,
    SourceHit,
    SubqueryResult,
)
from plugins.deep_research.synthesizer import Synthesizer


# ─── Test scaffolding ────────────────────────────────────────────────


class _ScriptedLLM:
    """Deterministic LLM stand-in: returns one response per call.

    The router lets a test target specific responses by ``brain`` (e.g.
    a4b for planner+synth, e4b for extractor) when the test cares about
    which call happens when.
    """

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        by_brain: dict[str, list[str]] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.by_brain: dict[str, list[str]] = {
            k: list(v) for k, v in (by_brain or {}).items()
        }
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        brain = kwargs.get("brain")
        if brain in self.by_brain and self.by_brain[brain]:
            return self.by_brain[brain].pop(0)
        if self.responses:
            return self.responses.pop(0)
        return ""


class _FakeToolRunner:
    """Stand-in for ``api.call_tool``. Routes by tool name."""

    def __init__(
        self,
        *,
        search_results: list[dict[str, Any]] | None = None,
        page_contents: dict[str, str] | None = None,
        search_error: str = "",
        fetch_error: str = "",
    ) -> None:
        self.search_results = list(search_results or [])
        self.page_contents = dict(page_contents or {})
        self.search_error = search_error
        self.fetch_error = fetch_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(args)))
        if name == "web_search":
            if self.search_error:
                return {"ok": False, "error": self.search_error}
            return {"ok": True, "data": {"results": list(self.search_results)}}
        if name == "web_fetch":
            url = args.get("url")
            if self.fetch_error:
                return {"ok": False, "error": self.fetch_error}
            content = self.page_contents.get(url)
            if content is None:
                return {"ok": False, "error": f"no content for {url}"}
            return {
                "ok": True,
                "data": {"url": url, "title": f"Title of {url}", "content": content},
            }
        return {"ok": False, "error": f"unknown tool: {name}"}


# ─── Plan parser ─────────────────────────────────────────────────────


class TestPlanParser:
    def test_strict_json_array(self) -> None:
        assert parse_subqueries('["a", "b", "c"]') == ["a", "b", "c"]

    def test_code_fence_stripped(self) -> None:
        assert parse_subqueries(
            '```json\n["alpha", "beta"]\n```'
        ) == ["alpha", "beta"]

    def test_numbered_list(self) -> None:
        assert parse_subqueries("1. first\n2. second") == ["first", "second"]

    def test_bullet_list(self) -> None:
        assert parse_subqueries("- one\n- two\n- three") == ["one", "two", "three"]

    def test_drops_quotes_around_items(self) -> None:
        # LLMs sometimes wrap items in quotes even in a numbered list.
        assert parse_subqueries('1. "first"\n2. "second"') == ["first", "second"]

    def test_drops_trailing_commas(self) -> None:
        assert parse_subqueries("1. first,\n2. second,") == ["first", "second"]

    def test_extracts_array_substring_from_prose(self) -> None:
        text = 'Here is the plan: ["q1", "q2"] — that is all.'
        assert parse_subqueries(text) == ["q1", "q2"]

    def test_empty_input(self) -> None:
        assert parse_subqueries("") == []
        assert parse_subqueries("   ") == []


class TestPlanner:
    @pytest.mark.asyncio
    async def test_returns_clean_unique_list(self) -> None:
        llm = _ScriptedLLM(responses=['["alpha", "beta", "Alpha", "ab"]'])
        # "Alpha" duplicates "alpha" (case-insensitive); "ab" is shorter
        # than min_query_len → filtered.
        planner = Planner(
            llm_chat=llm, min_subqueries=1, max_subqueries=10,
            default_subqueries=4,
        )
        plan = await planner.plan(topic="Test", language="de")
        assert plan == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_caps_at_max_subqueries(self) -> None:
        # Each item must clear _MIN_QUERY_LEN (4) so the parser doesn't
        # filter them out — that's a separate guard tested elsewhere.
        big = json.dumps([f"query number {i}" for i in range(20)])
        llm = _ScriptedLLM(responses=[big])
        planner = Planner(llm_chat=llm, max_subqueries=5)
        plan = await planner.plan(topic="x")
        assert len(plan) == 5

    @pytest.mark.asyncio
    async def test_falls_back_to_topic_on_empty_response(self) -> None:
        llm = _ScriptedLLM(responses=[""])
        planner = Planner(llm_chat=llm)
        plan = await planner.plan(topic="my topic")
        assert plan == ["my topic"]

    @pytest.mark.asyncio
    async def test_falls_back_to_topic_on_llm_exception(self) -> None:
        llm = _ScriptedLLM(raises=RuntimeError("boom"))
        planner = Planner(llm_chat=llm)
        plan = await planner.plan(topic="my topic")
        assert plan == ["my topic"]

    @pytest.mark.asyncio
    async def test_target_count_clamped(self) -> None:
        llm = _ScriptedLLM(responses=['["alpha", "beta", "gamma"]'])
        planner = Planner(llm_chat=llm, min_subqueries=2, max_subqueries=4)
        plan = await planner.plan(topic="x", target_count=99)
        # target_count above max must clamp without crashing; the parser
        # returned 3 valid entries, all kept.
        assert plan == ["alpha", "beta", "gamma"]


# ─── Extract parser + Researcher ─────────────────────────────────────


class TestExtractParser:
    def test_strict_json(self) -> None:
        rel, quotes = parse_extract('{"relevance": 0.8, "quotes": ["q1", "q2"]}')
        assert rel == 0.8
        assert quotes == ["q1", "q2"]

    def test_with_code_fence(self) -> None:
        rel, quotes = parse_extract(
            '```json\n{"relevance": 0.4, "quotes": []}\n```'
        )
        assert rel == 0.4
        assert quotes == []

    def test_garbage_yields_zeros(self) -> None:
        rel, quotes = parse_extract("not json")
        assert rel == 0.0
        assert quotes == []

    def test_numeric_relevance_coerced(self) -> None:
        rel, _ = parse_extract('{"relevance": "0.6", "quotes": []}')
        assert rel == 0.6

    def test_long_quote_truncated(self) -> None:
        long = "x" * 1000
        _, quotes = parse_extract(json.dumps({"relevance": 1, "quotes": [long]}))
        assert len(quotes) == 1
        assert quotes[0].endswith("…")
        assert len(quotes[0]) <= 601

    def test_alternative_key_snippets(self) -> None:
        # Some LLM rephrasings use "snippets" instead of "quotes".
        rel, quotes = parse_extract('{"relevance": 0.5, "snippets": ["a"]}')
        assert quotes == ["a"]


class TestResearcher:
    @pytest.mark.asyncio
    async def test_happy_path_one_query(self) -> None:
        runner = _FakeToolRunner(
            search_results=[
                {"url": "https://a.example", "title": "A", "snippet": "bla"},
                {"url": "https://b.example", "title": "B", "snippet": "blub"},
            ],
            page_contents={
                "https://a.example": "Contents of A page",
                "https://b.example": "Contents of B page",
            },
        )
        llm = _ScriptedLLM(
            responses=[
                # extractor for url a
                '{"relevance": 0.9, "quotes": ["fact about A"]}',
                # extractor for url b
                '{"relevance": 0.6, "quotes": ["fact about B"]}',
            ]
        )
        r = Researcher(
            call_tool=runner, llm_chat=llm,
            search_results_per_query=4, fetch_per_query=4,
        )
        result = await r.run(query="alpha", topic="alpha topic")
        assert result.completed
        assert len(result.sources) == 2
        # Both should be fetched + scored.
        for s in result.sources:
            assert s.fetched
            assert s.relevance > 0
            assert s.extracted

    @pytest.mark.asyncio
    async def test_search_error_short_circuits(self) -> None:
        runner = _FakeToolRunner(search_error="searxng down")
        llm = _ScriptedLLM(responses=[])
        r = Researcher(call_tool=runner, llm_chat=llm)
        result = await r.run(query="alpha")
        assert not result.completed
        assert "search_failed" in result.error
        assert llm.calls == []  # extractor not invoked

    @pytest.mark.asyncio
    async def test_fetch_error_marks_source_unfetched(self) -> None:
        runner = _FakeToolRunner(
            search_results=[{"url": "https://x.example", "title": "X"}],
            fetch_error="boom",
        )
        llm = _ScriptedLLM(responses=[])
        r = Researcher(call_tool=runner, llm_chat=llm)
        result = await r.run(query="alpha")
        assert result.completed
        assert len(result.sources) == 1
        s = result.sources[0]
        assert not s.fetched
        assert "fetch_failed" in s.fetch_error

    @pytest.mark.asyncio
    async def test_low_relevance_filtered_for_synthesis(self) -> None:
        # Set up two sources, one above the threshold and one below.
        runner = _FakeToolRunner(
            search_results=[
                {"url": "https://hi.example", "title": "Hi"},
                {"url": "https://lo.example", "title": "Lo"},
            ],
            page_contents={
                "https://hi.example": "high signal page",
                "https://lo.example": "low signal page",
            },
        )
        llm = _ScriptedLLM(
            responses=[
                '{"relevance": 0.9, "quotes": ["good"]}',
                '{"relevance": 0.1, "quotes": ["meh"]}',
            ]
        )
        r = Researcher(call_tool=runner, llm_chat=llm)
        result = await r.run(query="alpha")
        kept = relevant_sources(result)
        assert len(kept) == 1
        assert kept[0].url == "https://hi.example"

    @pytest.mark.asyncio
    async def test_emits_progress_events(self) -> None:
        runner = _FakeToolRunner(
            search_results=[{"url": "https://x.example", "title": "X"}],
            page_contents={"https://x.example": "page text"},
        )
        llm = _ScriptedLLM(responses=['{"relevance": 0.7, "quotes": ["q"]}'])
        r = Researcher(call_tool=runner, llm_chat=llm)
        events: list[dict[str, Any]] = []
        async def emit(payload):
            events.append(payload)
        await r.run(query="alpha", on_event=emit)
        kinds = [e.get("phase") for e in events]
        assert "search_done" in kinds
        assert "source_processed" in kinds


# ─── Synthesizer ─────────────────────────────────────────────────────


class TestSynthesizer:
    @pytest.mark.asyncio
    async def test_renders_with_citations(self) -> None:
        # Fake LLM returns a markdown report referencing sources [1] and [2].
        llm = _ScriptedLLM(responses=[
            "## Zusammenfassung\nKurz und knackig [1].\n\n"
            "## Sub-Frage A\nDetails [1].\n\n"
            "## Sub-Frage B\nMehr Details [2].\n\n"
            "## Quellen\n[1] First — https://first.example\n"
            "[2] Second — https://second.example"
        ])
        synth = Synthesizer(llm_chat=llm)
        task = ResearchTask(research_id="t1", topic="big topic", language="de")
        task.plan = ["alpha", "beta"]
        sr1 = SubqueryResult(query="alpha", completed=True)
        sr1.sources.append(SourceHit(
            url="https://first.example", title="First",
            subquery="alpha", fetched=True, relevance=0.8,
            extracted=["fact a"],
        ))
        sr2 = SubqueryResult(query="beta", completed=True)
        sr2.sources.append(SourceHit(
            url="https://second.example", title="Second",
            subquery="beta", fetched=True, relevance=0.7,
            extracted=["fact b"],
        ))
        task.subquery_results = [sr1, sr2]

        report = await synth.synthesise(task)
        assert "[1]" in report
        assert "[2]" in report
        assert "https://first.example" in report or "First" in report
        assert "## Quellen" in report

    @pytest.mark.asyncio
    async def test_no_relevant_sources_uses_fallback(self) -> None:
        llm = _ScriptedLLM(responses=["should not be called"])
        synth = Synthesizer(llm_chat=llm)
        task = ResearchTask(research_id="t1", topic="empty topic")
        task.plan = ["q1", "q2"]
        # All sub-queries have either no sources or low relevance.
        sr = SubqueryResult(query="q1", completed=True)
        sr.sources.append(
            SourceHit(url="https://a", relevance=0.05, fetched=True, extracted=[])
        )
        task.subquery_results = [sr]

        report = await synth.synthesise(task)
        # LLM was NOT called — pure fallback path.
        assert llm.calls == []
        assert "keine relevanten quellen" in report.lower()
        # The fallback lists the attempted sub-queries.
        for q in task.plan:
            assert q in report

    @pytest.mark.asyncio
    async def test_llm_failure_uses_deterministic_fallback(self) -> None:
        llm = _ScriptedLLM(raises=RuntimeError("synth boom"))
        synth = Synthesizer(llm_chat=llm)
        task = ResearchTask(research_id="t1", topic="topic")
        task.plan = ["q1"]
        sr = SubqueryResult(query="q1", completed=True)
        sr.sources.append(SourceHit(
            url="https://x.example", title="X",
            subquery="q1", fetched=True, relevance=0.9,
            extracted=["raw quote"],
        ))
        task.subquery_results = [sr]
        report = await synth.synthesise(task)
        # Fallback puts the URL + quote in plain markdown.
        assert "X" in report or "x.example" in report
        assert "raw quote" in report
        assert "Quellen" in report  # appendix always present

    @pytest.mark.asyncio
    async def test_appends_sources_section_when_llm_omits_it(self) -> None:
        llm = _ScriptedLLM(responses=[
            "## Zusammenfassung\nKurz [1]."
            # no "## Quellen" section!
        ])
        synth = Synthesizer(llm_chat=llm)
        task = ResearchTask(research_id="t1", topic="x")
        task.plan = ["q1"]
        sr = SubqueryResult(query="q1", completed=True)
        sr.sources.append(SourceHit(
            url="https://x.example", title="X",
            subquery="q1", fetched=True, relevance=0.9,
            extracted=["fact"],
        ))
        task.subquery_results = [sr]
        report = await synth.synthesise(task)
        assert "## Quellen" in report
        assert "https://x.example" in report

    @pytest.mark.asyncio
    async def test_dedup_sources_across_subqueries(self) -> None:
        # Same URL appears under two sub-queries — should get a single
        # citation number, with quotes merged.
        llm = _ScriptedLLM(responses=["## Quellen\n[1] X — https://x.example"])
        synth = Synthesizer(llm_chat=llm)
        task = ResearchTask(research_id="t1", topic="x")
        task.plan = ["q1", "q2"]
        for q in ("q1", "q2"):
            sr = SubqueryResult(query=q, completed=True)
            sr.sources.append(SourceHit(
                url="https://x.example", title="X",
                subquery=q, fetched=True, relevance=0.8,
                extracted=[f"fact from {q}"],
            ))
            task.subquery_results.append(sr)
        report = await synth.synthesise(task)
        # Only one [1] entry should be in the appendix.
        assert report.count("[1]") >= 1
        assert "[2]" not in report


# ─── State model ────────────────────────────────────────────────────


class TestStateModel:
    def test_progress_counters(self) -> None:
        task = ResearchTask(research_id="t1", topic="x")
        sr = SubqueryResult(query="q1", completed=True)
        sr.sources.extend([
            SourceHit(url="a", fetched=True, relevance=0.9, extracted=["q"]),
            SourceHit(url="b", fetched=True, relevance=0.1),  # low relevance, no quotes
            SourceHit(url="c", fetched=False),
        ])
        task.subquery_results.append(sr)
        prog = task.progress()
        assert prog["subqueries_completed"] == 1
        assert prog["subqueries_total"] == 1
        assert prog["sources_total"] == 3
        assert prog["sources_fetched"] == 2
        assert prog["sources_relevant"] == 1

    def test_to_public_roundtrip(self) -> None:
        task = ResearchTask(research_id="t1", topic="x")
        task.plan = ["a", "b"]
        task.state = "searching"
        public = task.to_public()
        assert public["research_id"] == "t1"
        assert public["state"] == "searching"
        assert public["plan"] == ["a", "b"]
        assert "progress" in public

    def test_source_hit_to_public_clamps_relevance(self) -> None:
        s = SourceHit(url="x", relevance=1.234567)
        out = s.to_public()
        # Round to 3 decimals.
        assert out["relevance"] == 1.235
