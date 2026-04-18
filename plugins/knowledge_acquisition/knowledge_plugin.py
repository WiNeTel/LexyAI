"""
Lexy AI - Knowledge Acquisition Plugin.

Autonomous RAG system that researches the internet, processes content,
and stores it categorized for LoRA training data generation.

Pipeline per job:
    search -> fetch -> chunk -> dedup -> categorize -> score -> store

All heavy lifting is delegated to the pipeline modules in
``plugins/knowledge_acquisition/pipeline/``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

from .exporter import TrainingDataExporter
from .pipeline.categorizer import ContentCategorizer
from .pipeline.chunker import ContentChunker
from .pipeline.crawler import KnowledgeCrawler
from .pipeline.deduplicator import ContentDeduplicator
from .pipeline.quality_scorer import QualityScorer

log = get_logger(module="knowledge_acquisition")

# ── Tool schemas ────────────────────────────────────────────────────────

RESEARCH_TOPIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "The topic to research on the web",
        },
        "max_pages": {
            "type": "integer",
            "description": "Maximum number of pages to crawl (default: config value)",
        },
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Override category list for classification",
        },
    },
    "required": ["topic"],
}

LIST_KNOWLEDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Semantic search query",
        },
        "category": {
            "type": "string",
            "description": "Filter by category",
        },
        "limit": {
            "type": "integer",
            "description": "Max results to return (default 10)",
        },
    },
}

KNOWLEDGE_STATS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

EXPORT_TRAINING_DATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": "Only export this category (default: all)",
        },
        "min_quality": {
            "type": "integer",
            "description": "Minimum quality score 1-5 (default: config value)",
        },
        "format": {
            "type": "string",
            "enum": ["jsonl", "alpaca"],
            "description": "Output format (default: jsonl)",
        },
    },
}


# ── Plugin ──────────────────────────────────────────────────────────────


class KnowledgeAcquisitionPlugin(BasePlugin):
    """Autonomous RAG: search, chunk, categorize, embed, store for LoRA training."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)

        # Config (gesetzt in on_load)
        self._max_concurrent_jobs: int = 2
        self._rate_limit_seconds: float = 3.0
        self._chunk_size: int = 800
        self._chunk_overlap: int = 100
        self._min_quality_score: int = 3
        self._dedup_threshold: float = 0.88
        self._max_pages_per_job: int = 10
        self._export_path: str = "./data/knowledge/exports"
        self._categories: list[str] = ["general"]

        # Pipeline Komponenten
        self._crawler: KnowledgeCrawler | None = None
        self._chunker: ContentChunker = ContentChunker()
        self._categorizer: ContentCategorizer = ContentCategorizer()
        self._deduplicator: ContentDeduplicator = ContentDeduplicator()
        self._scorer: QualityScorer = QualityScorer()
        self._exporter: TrainingDataExporter = TrainingDataExporter()

        # Job management
        self._job_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._running: bool = False
        self._workers: list[asyncio.Task[None]] = []
        self._active_jobs: dict[str, dict[str, Any]] = {}

    # ── Lifecycle ───────────────────────────────────────────────────

    async def on_load(self) -> None:
        """Initialise SQLite tables, config, export directory."""
        config = self.api.get_config()
        self._max_concurrent_jobs = int(config.get("max_concurrent_jobs", 2))
        self._rate_limit_seconds = float(config.get("rate_limit_seconds", 3.0))
        self._chunk_size = int(config.get("chunk_size", 800))
        self._chunk_overlap = int(config.get("chunk_overlap", 100))
        self._min_quality_score = int(config.get("min_quality_score", 3))
        self._dedup_threshold = float(config.get("dedup_threshold", 0.88))
        self._max_pages_per_job = int(config.get("max_pages_per_job", 10))
        self._export_path = str(config.get("export_path", "./data/knowledge/exports"))
        self._categories = list(config.get("categories", ["general"]))

        # Export-Verzeichnis erstellen
        Path(self._export_path).mkdir(parents=True, exist_ok=True)

        # SQLite Tabellen
        db = await self.api.get_db()
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                pages_found INTEGER DEFAULT 0,
                pages_fetched INTEGER DEFAULT 0,
                chunks_stored INTEGER DEFAULT 0,
                chunks_skipped INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                content_length INTEGER DEFAULT 0,
                chunks_created INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                quality_score INTEGER DEFAULT 0,
                source_url TEXT DEFAULT '',
                source_title TEXT DEFAULT '',
                chromadb_id TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (source_id) REFERENCES sources(id),
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);
            CREATE INDEX IF NOT EXISTS idx_chunks_quality ON chunks(quality_score);
            CREATE INDEX IF NOT EXISTS idx_chunks_job ON chunks(job_id);
            CREATE INDEX IF NOT EXISTS idx_sources_job ON sources(job_id);
            """
        )
        await db.commit()

        log.info(
            "knowledge.loaded",
            chunk_size=self._chunk_size,
            categories=len(self._categories),
            export_path=self._export_path,
        )

    async def on_enable(self) -> None:
        """Register tools, WS handlers, events, start worker loops."""
        # web_crawler Plugin holen
        web_crawler = self.api.get_plugin("web_crawler")
        if web_crawler is None:
            log.error("knowledge.missing_dependency", plugin="web_crawler")
            raise RuntimeError(
                "knowledge_acquisition requires the web_crawler plugin"
            )
        self._crawler = KnowledgeCrawler(web_crawler)

        # Tools registrieren
        self.api.register_tool(
            name="research_topic",
            handler=self._tool_research_topic,
            description=(
                "Research a topic on the web: search, fetch pages, chunk content, "
                "categorize and store in the knowledge base for later LoRA training. "
                "Returns a job ID for tracking progress."
            ),
            schema=RESEARCH_TOPIC_SCHEMA,
        )
        self.api.register_tool(
            name="list_knowledge",
            handler=self._tool_list_knowledge,
            description=(
                "Search the acquired knowledge base. Returns matching chunks "
                "with metadata (category, quality, source URL)."
            ),
            schema=LIST_KNOWLEDGE_SCHEMA,
        )
        self.api.register_tool(
            name="knowledge_stats",
            handler=self._tool_knowledge_stats,
            description=(
                "Show statistics about the knowledge base: chunk counts by "
                "category, recent jobs, total storage."
            ),
            schema=KNOWLEDGE_STATS_SCHEMA,
        )
        self.api.register_tool(
            name="export_training_data",
            handler=self._tool_export_training_data,
            description=(
                "Export stored knowledge chunks as JSONL or Alpaca format "
                "for LoRA fine-tuning. Generates instruction-response pairs."
            ),
            schema=EXPORT_TRAINING_DATA_SCHEMA,
        )

        # WS Handler registrieren
        self.api.register_ws_handler("knowledge_start_job", self._ws_start_job)
        self.api.register_ws_handler("knowledge_list_jobs", self._ws_list_jobs)
        self.api.register_ws_handler("knowledge_cancel_job", self._ws_cancel_job)

        # Worker-Loops starten
        self._running = True
        for i in range(self._max_concurrent_jobs):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"knowledge_worker_{i}",
            )
            self._workers.append(task)

        log.info(
            "knowledge.enabled",
            tools=["research_topic", "list_knowledge", "knowledge_stats", "export_training_data"],
            workers=self._max_concurrent_jobs,
        )

    async def on_disable(self) -> None:
        """Cancel running jobs, stop workers."""
        self._running = False

        # Aktive Jobs als abgebrochen markieren
        for job_id, job_info in self._active_jobs.items():
            log.info("knowledge.cancel_job", job_id=job_id)
            await self._update_job_status(job_id, "cancelled")

        # Worker-Tasks canceln
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._active_jobs.clear()

        # Queue leeren
        while not self._job_queue.empty():
            try:
                self._job_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        log.info("knowledge.disabled")

    # ── Worker Loop ────────────────────────────────────────────────

    async def _worker_loop(self, worker_id: int) -> None:
        """Pull jobs from queue and run the full pipeline."""
        log.info("knowledge.worker_started", worker_id=worker_id)
        while self._running:
            try:
                job_data = await asyncio.wait_for(
                    self._job_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            job_id = job_data["job_id"]
            log.info(
                "knowledge.worker_picked_job",
                worker_id=worker_id,
                job_id=job_id,
                topic=job_data["topic"],
            )

            try:
                await self._run_job(
                    job_id=job_id,
                    topic=job_data["topic"],
                    max_pages=job_data["max_pages"],
                    categories=job_data["categories"],
                )
            except asyncio.CancelledError:
                await self._update_job_status(job_id, "cancelled")
                break
            except Exception as exc:
                log.error(
                    "knowledge.job_failed",
                    job_id=job_id,
                    error=str(exc),
                    exc_info=True,
                )
                await self._update_job_status(
                    job_id, "failed", error=str(exc)
                )
            finally:
                self._active_jobs.pop(job_id, None)

        log.info("knowledge.worker_stopped", worker_id=worker_id)

    # ── Full Pipeline ──────────────────────────────────────────────

    async def _run_job(
        self,
        job_id: str,
        topic: str,
        max_pages: int,
        categories: list[str],
    ) -> None:
        """Execute the full knowledge acquisition pipeline for one job."""
        assert self._crawler is not None  # noqa: S101

        self._active_jobs[job_id] = {"topic": topic, "status": "running"}

        # 1. Status auf 'running' setzen
        await self._update_job_status(job_id, "running")
        await self._broadcast_progress(job_id, "running", "Searching...")

        # 2. Suche ausfuehren
        search_results = await self._crawler.search_topic(topic, max_pages)
        pages_found = len(search_results)
        await self._update_job_field(job_id, "pages_found", pages_found)

        if not search_results:
            await self._update_job_status(job_id, "done", error="No results found")
            await self._broadcast_progress(job_id, "done", "No results found")
            return

        await self._broadcast_progress(
            job_id, "running", f"Found {pages_found} pages, fetching..."
        )

        # 3. Jede URL verarbeiten (rate-limited)
        total_chunks_stored = 0
        total_chunks_skipped = 0
        pages_fetched = 0
        db = await self.api.get_db()

        for idx, result in enumerate(search_results):
            if not self._running:
                log.info("knowledge.job_interrupted", job_id=job_id)
                break

            url = result["url"]
            title = result.get("title", "")
            source_id = uuid.uuid4().hex

            # Source-Record erstellen
            now = time.time()
            await db.execute(
                "INSERT INTO sources (id, job_id, url, title, status, created_at) "
                "VALUES (?, ?, ?, ?, 'fetching', ?)",
                (source_id, job_id, url, title, now),
            )
            await db.commit()

            # 3a. Seite abrufen
            page = await self._crawler.fetch_page(url)
            if page is None:
                await db.execute(
                    "UPDATE sources SET status='failed', error='fetch_failed' WHERE id=?",
                    (source_id,),
                )
                await db.commit()
                continue

            pages_fetched += 1
            content = page["content"]
            page_title = page.get("title", title)

            await db.execute(
                "UPDATE sources SET status='processing', title=?, content_length=? WHERE id=?",
                (page_title, len(content), source_id),
            )
            await db.commit()

            # 3b. Content in Chunks aufteilen
            chunks = self._chunker.chunk(
                content,
                chunk_size=self._chunk_size,
                overlap=self._chunk_overlap,
            )

            source_chunks_stored = 0

            # 3c. Jeden Chunk verarbeiten
            for chunk_text in chunks:
                if not self._running:
                    break

                # Zu kurze Chunks ueberspringen
                if len(chunk_text.strip()) < 50:
                    total_chunks_skipped += 1
                    continue

                # Dedup-Check
                is_dup = await self._deduplicator.is_duplicate(
                    chunk_text, self._dedup_threshold, self.api
                )
                if is_dup:
                    total_chunks_skipped += 1
                    continue

                # Kategorisieren
                category = await self._categorizer.categorize(
                    chunk_text, categories, self.api
                )

                # Qualitaet bewerten
                quality = await self._scorer.score(chunk_text, self.api)
                if quality < self._min_quality_score:
                    total_chunks_skipped += 1
                    continue

                # In ChromaDB speichern
                chunk_id = uuid.uuid4().hex
                chromadb_id = ""
                try:
                    chromadb_id = await self.api.memory_store(
                        text=chunk_text,
                        collection="knowledge",
                        metadata={
                            "category": category,
                            "quality_score": quality,
                            "source_url": url,
                            "source_title": page_title,
                            "job_id": job_id,
                            "chunk_id": chunk_id,
                        },
                    )
                except Exception as exc:
                    log.warning(
                        "knowledge.chromadb_store_error",
                        chunk_id=chunk_id,
                        error=str(exc),
                    )

                # In SQLite speichern
                now = time.time()
                await db.execute(
                    "INSERT INTO chunks "
                    "(id, source_id, job_id, content, category, quality_score, "
                    "source_url, source_title, chromadb_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id, source_id, job_id, chunk_text,
                        category, quality, url, page_title,
                        chromadb_id, now,
                    ),
                )
                total_chunks_stored += 1
                source_chunks_stored += 1

            await db.commit()

            # Source-Record aktualisieren
            await db.execute(
                "UPDATE sources SET status='done', chunks_created=? WHERE id=?",
                (source_chunks_stored, source_id),
            )
            await db.commit()

            # Job-Zaehler aktualisieren
            await db.execute(
                "UPDATE jobs SET pages_fetched=?, chunks_stored=?, chunks_skipped=?, updated_at=? "
                "WHERE id=?",
                (pages_fetched, total_chunks_stored, total_chunks_skipped, time.time(), job_id),
            )
            await db.commit()

            # Fortschritt broadcasten
            await self._broadcast_progress(
                job_id,
                "running",
                f"Page {idx + 1}/{pages_found}: {source_chunks_stored} chunks stored",
                pages_fetched=pages_fetched,
                pages_found=pages_found,
                chunks_stored=total_chunks_stored,
            )

            # Rate-Limiting zwischen Seiten
            if idx < len(search_results) - 1:
                await asyncio.sleep(self._rate_limit_seconds)

        # 4. Job abschliessen
        final_status = "done" if self._running else "cancelled"
        await self._update_job_status(job_id, final_status)
        await self._broadcast_progress(
            job_id,
            final_status,
            f"Completed: {total_chunks_stored} chunks stored, {total_chunks_skipped} skipped",
            pages_fetched=pages_fetched,
            pages_found=pages_found,
            chunks_stored=total_chunks_stored,
        )

        # Event emittieren
        await self.api.emit(
            "knowledge.job_complete",
            {
                "job_id": job_id,
                "topic": topic,
                "status": final_status,
                "pages_fetched": pages_fetched,
                "chunks_stored": total_chunks_stored,
                "chunks_skipped": total_chunks_skipped,
            },
        )

        log.info(
            "knowledge.job_complete",
            job_id=job_id,
            topic=topic,
            status=final_status,
            pages_fetched=pages_fetched,
            chunks_stored=total_chunks_stored,
            chunks_skipped=total_chunks_skipped,
        )

    # ── Tools ──────────────────────────────────────────────────────

    async def _tool_research_topic(
        self,
        topic: str,
        max_pages: int | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a research job and enqueue it."""
        job_id = uuid.uuid4().hex
        now = time.time()
        effective_max_pages = max_pages or self._max_pages_per_job
        effective_categories = categories or self._categories

        # Job in DB erstellen
        db = await self.api.get_db()
        await db.execute(
            "INSERT INTO jobs (id, topic, status, created_at, updated_at) "
            "VALUES (?, ?, 'pending', ?, ?)",
            (job_id, topic, now, now),
        )
        await db.commit()

        # Job in die Queue legen
        await self._job_queue.put(
            {
                "job_id": job_id,
                "topic": topic,
                "max_pages": effective_max_pages,
                "categories": effective_categories,
            }
        )

        log.info(
            "knowledge.job_created",
            job_id=job_id,
            topic=topic,
            max_pages=effective_max_pages,
        )

        return {
            "job_id": job_id,
            "topic": topic,
            "max_pages": effective_max_pages,
            "categories": effective_categories,
            "status": "pending",
            "message": f"Research job created for '{topic}'. Processing in background.",
        }

    async def _tool_list_knowledge(
        self,
        query: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Query knowledge base via ChromaDB + SQLite."""
        effective_limit = min(limit or 10, 50)

        results: list[dict[str, Any]] = []

        # ChromaDB semantische Suche wenn Query vorhanden
        if query:
            memory_results = await self.api.memory_recall(
                query=query,
                collection="knowledge",
                limit=effective_limit,
            )
            for item in memory_results:
                meta = item.get("metadata", {})
                # Kategorie-Filter anwenden
                if category and meta.get("category", "") != category:
                    continue
                results.append(
                    {
                        "text": item.get("text", "")[:300],
                        "category": meta.get("category", "unknown"),
                        "quality_score": meta.get("quality_score", 0),
                        "source_url": meta.get("source_url", ""),
                        "source_title": meta.get("source_title", ""),
                        "score": round(item.get("score", 0.0), 3),
                    }
                )
        else:
            # Ohne Query: SQLite-basierte Auflistung
            db = await self.api.get_db()
            sql = "SELECT content, category, quality_score, source_url, source_title FROM chunks"
            params: list[Any] = []
            if category:
                sql += " WHERE category = ?"
                params.append(category)
            sql += " ORDER BY quality_score DESC, created_at DESC LIMIT ?"
            params.append(effective_limit)

            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                results.append(
                    {
                        "text": row[0][:300],
                        "category": row[1],
                        "quality_score": row[2],
                        "source_url": row[3],
                        "source_title": row[4],
                    }
                )

        return {
            "query": query,
            "category": category,
            "count": len(results),
            "results": results,
        }

    async def _tool_knowledge_stats(self) -> dict[str, Any]:
        """Aggregate counts by category, recent jobs."""
        db = await self.api.get_db()

        # Chunks pro Kategorie
        async with db.execute(
            "SELECT category, COUNT(*), AVG(quality_score) "
            "FROM chunks GROUP BY category ORDER BY COUNT(*) DESC"
        ) as cursor:
            cat_rows = await cursor.fetchall()

        categories_stats: list[dict[str, Any]] = []
        total_chunks = 0
        for row in cat_rows:
            count = row[1]
            total_chunks += count
            categories_stats.append(
                {
                    "category": row[0],
                    "count": count,
                    "avg_quality": round(row[2], 1) if row[2] else 0.0,
                }
            )

        # Letzte 10 Jobs
        async with db.execute(
            "SELECT id, topic, status, pages_found, pages_fetched, "
            "chunks_stored, chunks_skipped, created_at "
            "FROM jobs ORDER BY created_at DESC LIMIT 10"
        ) as cursor:
            job_rows = await cursor.fetchall()

        recent_jobs: list[dict[str, Any]] = []
        for row in job_rows:
            recent_jobs.append(
                {
                    "job_id": row[0],
                    "topic": row[1],
                    "status": row[2],
                    "pages_found": row[3],
                    "pages_fetched": row[4],
                    "chunks_stored": row[5],
                    "chunks_skipped": row[6],
                    "created_at": row[7],
                }
            )

        # Gesamtanzahl Sources
        async with db.execute("SELECT COUNT(*) FROM sources") as cursor:
            total_sources = (await cursor.fetchone())[0]  # type: ignore[index]

        return {
            "total_chunks": total_chunks,
            "total_sources": total_sources,
            "categories": categories_stats,
            "recent_jobs": recent_jobs,
            "active_jobs": list(self._active_jobs.keys()),
            "queue_size": self._job_queue.qsize(),
        }

    async def _tool_export_training_data(
        self,
        category: str | None = None,
        min_quality: int | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Generate JSONL or Alpaca training data export."""
        effective_quality = min_quality or self._min_quality_score
        export_format = format or "jsonl"
        db = await self.api.get_db()

        if export_format == "alpaca":
            result = await self._exporter.export_alpaca(
                db=db,
                export_path=self._export_path,
                category=category,
                min_quality=effective_quality,
                api=self.api,
            )
        else:
            result = await self._exporter.export_jsonl(
                db=db,
                export_path=self._export_path,
                category=category,
                min_quality=effective_quality,
                api=self.api,
            )

        return result

    # ── WS Handlers ────────────────────────────────────────────────

    async def _ws_start_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """WS handler: start a research job from frontend."""
        topic = data.get("topic", "").strip()
        if not topic:
            return {"error": "topic is required"}

        max_pages = data.get("max_pages")
        categories = data.get("categories")

        result = await self._tool_research_topic(
            topic=topic,
            max_pages=max_pages,
            categories=categories,
        )
        return result

    async def _ws_list_jobs(self, data: dict[str, Any]) -> dict[str, Any]:
        """WS handler: list all jobs with status."""
        db = await self.api.get_db()
        limit = data.get("limit", 20)

        async with db.execute(
            "SELECT id, topic, status, pages_found, pages_fetched, "
            "chunks_stored, chunks_skipped, error, created_at, updated_at "
            "FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

        jobs: list[dict[str, Any]] = []
        for row in rows:
            jobs.append(
                {
                    "job_id": row[0],
                    "topic": row[1],
                    "status": row[2],
                    "pages_found": row[3],
                    "pages_fetched": row[4],
                    "chunks_stored": row[5],
                    "chunks_skipped": row[6],
                    "error": row[7],
                    "created_at": row[8],
                    "updated_at": row[9],
                }
            )

        return {"jobs": jobs, "active": list(self._active_jobs.keys())}

    async def _ws_cancel_job(self, data: dict[str, Any]) -> dict[str, Any]:
        """WS handler: cancel a running or pending job."""
        job_id = data.get("job_id", "").strip()
        if not job_id:
            return {"error": "job_id is required"}

        if job_id in self._active_jobs:
            # Aktiven Job als cancelled markieren — der Worker prueft self._running
            # bzw. bricht beim naechsten Schleifendurchlauf ab
            await self._update_job_status(job_id, "cancelled")
            self._active_jobs.pop(job_id, None)
            log.info("knowledge.job_cancelled", job_id=job_id)
            return {"job_id": job_id, "status": "cancelled"}

        # Job in DB suchen und Status pruefen
        db = await self.api.get_db()
        async with db.execute(
            "SELECT status FROM jobs WHERE id=?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return {"error": f"Job {job_id} not found"}

        if row[0] == "pending":
            await self._update_job_status(job_id, "cancelled")
            return {"job_id": job_id, "status": "cancelled"}

        return {"job_id": job_id, "status": row[0], "message": "Job already finished"}

    # ── Interne Helfer ─────────────────────────────────────────────

    async def _update_job_status(
        self, job_id: str, status: str, error: str | None = None
    ) -> None:
        """Update job status in SQLite."""
        db = await self.api.get_db()
        now = time.time()
        if error:
            await db.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error, now, job_id),
            )
        else:
            await db.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                (status, now, job_id),
            )
        await db.commit()

    async def _update_job_field(
        self, job_id: str, field: str, value: Any
    ) -> None:
        """Update a single numeric field on a job row."""
        # Nur bekannte Felder zulassen um SQL Injection zu verhindern
        allowed = {"pages_found", "pages_fetched", "chunks_stored", "chunks_skipped"}
        if field not in allowed:
            log.warning("knowledge.invalid_field", field=field)
            return
        db = await self.api.get_db()
        now = time.time()
        await db.execute(
            f"UPDATE jobs SET {field}=?, updated_at=? WHERE id=?",  # noqa: S608
            (value, now, job_id),
        )
        await db.commit()

    async def _broadcast_progress(
        self,
        job_id: str,
        status: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Broadcast job progress to all WS clients."""
        payload: dict[str, Any] = {
            "type": "knowledge_progress",
            "job_id": job_id,
            "status": status,
            "message": message,
            **extra,
        }
        try:
            await self.api.ws_broadcast(payload)
        except Exception as exc:
            log.debug("knowledge.broadcast_error", error=str(exc))
