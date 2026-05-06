"""
Lexy AI — Deep-Research plugin.

Composition root for the four-stage research pipeline. Public surface:

* ``deep_research({topic, max_subqueries?, language?})`` — runs the
  pipeline, returns the final report. Blocks for the duration of the
  research (typically 30–120 s) but emits live progress over WS so
  the frontend can render a research console while the LLM waits.
* ``deep_research_status({research_id})`` — current snapshot.
* ``deep_research_stop({research_id})`` — cancel an in-flight task.
* ``deep_research_list_tasks({})`` — recent + active tasks.

Pipeline:

    Plan (planner_brain) →
        Sub-queries [N] →
            ┌── Search + Fetch + Extract ─┐
            │  (parallel asyncio.gather)  │
            └─────────────────────────────┘
                ↓
            Synth (synth_brain, large)
                ↓
            Markdown report w/ citations
                ↓
            (optional) memory_store(category="research")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from lexy_core.plugin_system import BasePlugin

from .planner import Planner
from .researcher import Researcher
from .state import ResearchTask, SubqueryResult
from .synthesizer import Synthesizer


log = logging.getLogger(__name__)


# ─── Tool schemas ────────────────────────────────────────────────────


DEEP_RESEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": (
                "Was soll recherchiert werden? Ein konkreter Satz oder "
                "Stichwort, idealerweise so wie die Original-User-Frage."
            ),
        },
        "max_subqueries": {
            "type": "integer",
            "minimum": 2,
            "maximum": 10,
            "description": (
                "Wie viele Sub-Anfragen soll die Planung erzeugen? "
                "Default 5. Mehr = gründlicher, aber langsamer."
            ),
        },
        "language": {
            "type": "string",
            "description": "Sprache der Suche + des Reports. Default 'de'.",
        },
    },
    "required": ["topic"],
}

STATUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"research_id": {"type": "string"}},
    "required": ["research_id"],
}

STOP_SCHEMA = STATUS_SCHEMA
LIST_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


# ─── Plugin ──────────────────────────────────────────────────────────


class DeepResearchPlugin(BasePlugin):
    """Multi-step web research with citations."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._planner: Planner | None = None
        self._researcher: Researcher | None = None
        self._synth: Synthesizer | None = None

        self._tasks: dict[str, ResearchTask] = {}
        self._runners: dict[str, asyncio.Task[None]] = {}

        # Config (resolved in on_load)
        self._planner_brain: str = "a4b"
        self._planner_max_tokens: int = 600
        self._planner_temperature: float = 0.4
        self._min_subqueries: int = 3
        self._max_subqueries: int = 7
        self._default_subqueries: int = 5

        self._search_results_per_query: int = 8
        self._fetch_per_query: int = 4
        self._fetch_concurrency: int = 2
        self._fetch_timeout: float = 15.0
        self._extractor_brain: str = "e4b"
        self._extractor_max_tokens: int = 400
        self._extractor_temp: float = 0.2
        self._page_text_max: int = 6000

        self._synth_brain: str = "a4b"
        self._synth_max_tokens: int = 2400
        self._synth_temp: float = 0.5
        self._persist_to_memory: bool = True

        self._parallel_subqueries: int = 3
        self._total_timeout: float = 240.0
        self._default_language: str = "de"

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def on_load(self) -> None:
        cfg = self.api.get_config()

        # Planning
        self._planner_brain = str(cfg.get("planner_brain") or "a4b")
        self._planner_max_tokens = int(cfg.get("planner_max_tokens") or 600)
        self._planner_temperature = float(cfg.get("planner_temperature") or 0.4)
        self._min_subqueries = int(cfg.get("min_subqueries") or 3)
        self._max_subqueries = int(cfg.get("max_subqueries") or 7)
        self._default_subqueries = int(cfg.get("default_subqueries") or 5)

        # Search + extract
        self._search_results_per_query = int(cfg.get("search_results_per_query") or 8)
        self._fetch_per_query = int(cfg.get("fetch_per_query") or 4)
        self._fetch_concurrency = int(cfg.get("fetch_concurrency_per_query") or 2)
        self._fetch_timeout = float(cfg.get("fetch_timeout_seconds") or 15.0)
        self._extractor_brain = str(cfg.get("extractor_brain") or "e4b")
        self._extractor_max_tokens = int(cfg.get("extractor_max_tokens") or 400)
        self._extractor_temp = float(cfg.get("extractor_temperature") or 0.2)
        self._page_text_max = int(cfg.get("page_text_max_chars") or 6000)

        # Synth
        self._synth_brain = str(cfg.get("synth_brain") or "a4b")
        self._synth_max_tokens = int(cfg.get("synth_max_tokens") or 2400)
        self._synth_temp = float(cfg.get("synth_temperature") or 0.5)
        self._persist_to_memory = bool(cfg.get("persist_to_memory", True))

        # Pipeline-wide
        self._parallel_subqueries = int(cfg.get("parallel_subqueries") or 3)
        self._total_timeout = float(cfg.get("total_timeout_seconds") or 240.0)
        self._default_language = str(cfg.get("default_language") or "de")

        self._planner = Planner(
            llm_chat=self.api.llm_chat,
            brain=self._planner_brain,
            max_tokens=self._planner_max_tokens,
            temperature=self._planner_temperature,
            min_subqueries=self._min_subqueries,
            max_subqueries=self._max_subqueries,
            default_subqueries=self._default_subqueries,
        )
        self._researcher = Researcher(
            call_tool=self.api.call_tool,
            llm_chat=self.api.llm_chat,
            search_results_per_query=self._search_results_per_query,
            fetch_per_query=self._fetch_per_query,
            fetch_concurrency=self._fetch_concurrency,
            page_text_max_chars=self._page_text_max,
            extractor_brain=self._extractor_brain,
            extractor_max_tokens=self._extractor_max_tokens,
            extractor_temperature=self._extractor_temp,
        )
        self._synth = Synthesizer(
            llm_chat=self.api.llm_chat,
            brain=self._synth_brain,
            max_tokens=self._synth_max_tokens,
            temperature=self._synth_temp,
        )

        log.info(
            "deep_research.loaded planner=%s extractor=%s synth=%s "
            "parallel=%d timeout=%ds",
            self._planner_brain, self._extractor_brain, self._synth_brain,
            self._parallel_subqueries, int(self._total_timeout),
        )

    async def on_enable(self) -> None:
        api = self.api
        api.register_tool(
            "deep_research",
            self._tool_deep_research,
            description=(
                "Multi-Step-Recherche: zerlegt das Thema in Sub-Queries, "
                "sucht und liest mehrere Webseiten parallel, extrahiert "
                "relevante Zitate und erstellt einen strukturierten "
                "Markdown-Bericht mit Quellen-Citations [1], [2], … . "
                "Blockiert für ~30-120 s während die Pipeline läuft, "
                "sendet währenddessen Live-Progress an die Frontend-Konsole."
            ),
            schema=DEEP_RESEARCH_SCHEMA,
        )
        api.register_tool(
            "deep_research_status",
            self._tool_status,
            description="Aktueller Snapshot eines Research-Tasks (plan, sources, report).",
            schema=STATUS_SCHEMA,
        )
        api.register_tool(
            "deep_research_stop",
            self._tool_stop,
            description="Bricht einen laufenden Research-Task ab.",
            schema=STOP_SCHEMA,
        )
        api.register_tool(
            "deep_research_list_tasks",
            self._tool_list_tasks,
            description="Liste aller aktiven + jüngst beendeten Research-Tasks.",
            schema=LIST_SCHEMA,
        )

        api.register_ws_handler(
            "deep_research_status_query", self._ws_status_query,
        )

        log.info("deep_research.enabled tools=4")

    async def on_disable(self) -> None:
        # Cancel any in-flight tasks so they don't keep running.
        for tid in list(self._runners):
            await self.stop(tid)
        log.info("deep_research.disabled")

    # ─── Tool implementations ──────────────────────────────────────

    async def _tool_deep_research(
        self,
        topic: str,
        max_subqueries: int | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        if not topic or not topic.strip():
            return {"ok": False, "error": "topic_required"}
        if (
            self._planner is None
            or self._researcher is None
            or self._synth is None
        ):
            return {"ok": False, "error": "plugin_not_initialised"}

        task = ResearchTask(
            research_id=uuid.uuid4().hex[:12],
            topic=topic.strip(),
            language=(language or self._default_language).strip() or "de",
        )
        self._tasks[task.research_id] = task
        runner = asyncio.create_task(
            self._run_pipeline(task, target_count=max_subqueries),
            name=f"deep_research.{task.research_id}",
        )
        self._runners[task.research_id] = runner

        # Block (with WS progress emitting) until the runner finishes.
        try:
            await runner
        except asyncio.CancelledError:
            pass
        finally:
            self._runners.pop(task.research_id, None)

        result_payload: dict[str, Any] = {
            "ok": task.state == "done",
            "research_id": task.research_id,
            "state": task.state,
            "topic": task.topic,
            "plan": list(task.plan),
            "sources_total": task.progress()["sources_total"],
            "sources_relevant": task.progress()["sources_relevant"],
            "report": task.report,
        }
        if task.state == "failed":
            result_payload["error"] = task.last_error or "unknown_error"
        return result_payload

    async def _tool_status(self, research_id: str) -> dict[str, Any]:
        task = self._tasks.get(research_id)
        if task is None:
            return {"ok": False, "error": "not_found"}
        return {"ok": True, "task": task.to_public()}

    async def _tool_stop(self, research_id: str) -> dict[str, Any]:
        ok = await self.stop(research_id)
        return {"ok": ok}

    async def _tool_list_tasks(self) -> dict[str, Any]:
        return {
            "ok": True,
            "tasks": [t.to_public() for t in self._tasks.values()],
        }

    async def _ws_status_query(
        self, client: Any, message: dict[str, Any],
    ) -> None:
        rid = str((message or {}).get("research_id") or "")
        if rid:
            task = self._tasks.get(rid)
            await client.send_json(
                {
                    "type": "deep_research_status",
                    "ok": task is not None,
                    "task": task.to_public() if task else None,
                }
            )
            return
        await client.send_json(
            {
                "type": "deep_research_status",
                "ok": True,
                "tasks": [t.to_public() for t in self._tasks.values()],
            }
        )

    # ─── Pipeline ──────────────────────────────────────────────────

    async def stop(self, research_id: str) -> bool:
        runner = self._runners.get(research_id)
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass
        task = self._tasks.get(research_id)
        if task is not None and task.state not in ("done", "failed", "cancelled"):
            task.state = "cancelled"
            task.finished_at = time.time()
            await self._broadcast(task, "deep_research_cancelled")
        return runner is not None

    async def _run_pipeline(
        self, task: ResearchTask, *, target_count: int | None,
    ) -> None:
        """Execute plan → search → synth, with a wall-clock guard."""
        try:
            await asyncio.wait_for(
                self._pipeline_inner(task, target_count=target_count),
                timeout=self._total_timeout,
            )
        except asyncio.TimeoutError:
            log.warning(
                "deep_research.timeout id=%s after=%ss",
                task.research_id, int(self._total_timeout),
            )
            # Whatever sub-queries finished get synthesised; the rest
            # are recorded as incomplete. We still try to give Mike a
            # report instead of bailing.
            await self._safe_synth(task, partial=True)
        except asyncio.CancelledError:
            task.state = "cancelled"
            task.finished_at = time.time()
            await self._broadcast(task, "deep_research_cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("deep_research.pipeline_unhandled err=%s", exc)
            task.state = "failed"
            task.last_error = str(exc)[:400]
            task.finished_at = time.time()
            await self._broadcast(task, "deep_research_error")

    async def _pipeline_inner(
        self, task: ResearchTask, *, target_count: int | None,
    ) -> None:
        # ── 1) Plan ──────────────────────────────────────────────
        task.state = "planning"
        await self._broadcast(task, "deep_research_started")
        assert self._planner is not None
        plan = await self._planner.plan(
            topic=task.topic,
            language=task.language,
            target_count=target_count,
        )
        task.plan = plan
        task.subquery_results = [SubqueryResult(query=q) for q in plan]
        await self._broadcast(task, "deep_research_planned")

        # ── 2) Search + fetch + extract ──────────────────────────
        task.state = "searching"
        await self._broadcast(task, "deep_research_searching")
        assert self._researcher is not None
        sem = asyncio.Semaphore(max(1, self._parallel_subqueries))

        async def _run_one(sr_index: int, query: str) -> None:
            async with sem:
                async def emit_event(payload: dict[str, Any]) -> None:
                    await self.api.ws_broadcast(
                        {
                            "type": "deep_research_subquery_progress",
                            "research_id": task.research_id,
                            **payload,
                        }
                    )
                try:
                    sr = await self._researcher.run(
                        query=query,
                        topic=task.topic,
                        language=task.language,
                        on_event=emit_event,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    sr = SubqueryResult(query=query, error=str(exc)[:400])
                task.subquery_results[sr_index] = sr
                await self._broadcast(task, "deep_research_subquery_done")

        await asyncio.gather(
            *(
                _run_one(i, q) for i, q in enumerate(plan)
            ),
            return_exceptions=False,
        )

        # ── 3) Synthesise ────────────────────────────────────────
        await self._safe_synth(task, partial=False)

    async def _safe_synth(self, task: ResearchTask, *, partial: bool) -> None:
        if task.state in ("cancelled", "failed"):
            return
        task.state = "synthesising"
        await self._broadcast(task, "deep_research_synthesising")
        assert self._synth is not None
        report = await self._synth.synthesise(task)
        task.report = report
        task.state = "done"
        task.finished_at = time.time()
        await self._broadcast(task, "deep_research_done")

        if self._persist_to_memory and report:
            await self._persist_report(task)

    async def _persist_report(self, task: ResearchTask) -> None:
        try:
            await self.api.memory_store(
                text=task.report,
                collection="context",
                metadata={
                    "source": "deep_research",
                    "kind": "research_report",
                    "research_id": task.research_id,
                    "topic": task.topic,
                    "language": task.language,
                    "subqueries": list(task.plan),
                    "session_id": task.session_id or "",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("deep_research.memory_store_failed err=%s", exc)

    async def _broadcast(self, task: ResearchTask, event_type: str) -> None:
        try:
            await self.api.ws_broadcast(
                {"type": event_type, **task.to_public()}
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("deep_research.broadcast_failed err=%s", exc)
