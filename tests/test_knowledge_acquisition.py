"""Tests for the Knowledge Acquisition plugin.

Covers the pipeline components (chunker, categorizer, deduplicator,
quality scorer, exporter) and the main plugin tool handlers with
mocked dependencies.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from plugins.knowledge_acquisition.pipeline.chunker import ContentChunker
from plugins.knowledge_acquisition.pipeline.categorizer import ContentCategorizer
from plugins.knowledge_acquisition.pipeline.crawler import KnowledgeCrawler
from plugins.knowledge_acquisition.pipeline.deduplicator import ContentDeduplicator
from plugins.knowledge_acquisition.pipeline.quality_scorer import QualityScorer
from plugins.knowledge_acquisition.exporter import TrainingDataExporter


# ── ContentChunker ─────────────────────────────────────────────────────


class TestContentChunker:
    def setup_method(self) -> None:
        self.chunker = ContentChunker()

    def test_empty_input(self) -> None:
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   ") == []

    def test_small_text_single_chunk(self) -> None:
        text = "This is a short paragraph."
        chunks = self.chunker.chunk(text, chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_splitting(self) -> None:
        para1 = "A" * 300
        para2 = "B" * 300
        para3 = "C" * 300
        text = f"{para1}\n\n{para2}\n\n{para3}"
        chunks = self.chunker.chunk(text, chunk_size=400, overlap=0)
        # Paragraphen sollten einzeln sein da jeder 300 Zeichen hat
        # und 2 zusammen > 400 waeren
        assert len(chunks) >= 2

    def test_overlap_added(self) -> None:
        para1 = "First paragraph with some content here."
        para2 = "Second paragraph with different content."
        text = f"{para1}\n\n{para2}"
        chunks = self.chunker.chunk(text, chunk_size=50, overlap=10)
        if len(chunks) > 1:
            # Der zweite Chunk sollte Overlap-Text vom ersten enthalten
            assert len(chunks[1]) > len("Second paragraph with different content.")

    def test_long_paragraph_splits(self) -> None:
        # Ein einzelner langer Paragraph ohne Absaetze
        text = "word " * 500  # ~2500 Zeichen
        chunks = self.chunker.chunk(text.strip(), chunk_size=300, overlap=0)
        assert len(chunks) > 1
        for chunk in chunks:
            # Jeder Chunk sollte ungefaehr chunk_size sein (mit Toleranz)
            assert len(chunk) <= 400  # Etwas Toleranz fuer Wortgrenzen

    def test_sentence_boundary_splitting(self) -> None:
        text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. And this is the fourth one."
        )
        chunks = self.chunker.chunk(text, chunk_size=80, overlap=0)
        assert len(chunks) >= 2


# ── ContentCategorizer ─────────────────────────────────────────────────


class TestContentCategorizer:
    def setup_method(self) -> None:
        self.categorizer = ContentCategorizer()

    @pytest.mark.asyncio
    async def test_exact_match(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="python_docs")
        categories = ["python_docs", "tutorial", "general"]
        result = await self.categorizer.categorize("some python docs", categories, api)
        assert result == "python_docs"

    @pytest.mark.asyncio
    async def test_fallback_to_general(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="some_unknown_category")
        categories = ["python_docs", "tutorial", "general"]
        result = await self.categorizer.categorize("random text", categories, api)
        assert result == "general"

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        categories = ["python_docs", "tutorial", "general"]
        result = await self.categorizer.categorize("text", categories, api)
        assert result == "general"

    @pytest.mark.asyncio
    async def test_empty_categories(self) -> None:
        api = AsyncMock()
        result = await self.categorizer.categorize("text", [], api)
        assert result == "general"

    @pytest.mark.asyncio
    async def test_partial_match(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="Category: python_docs")
        categories = ["python_docs", "tutorial", "general"]
        result = await self.categorizer.categorize("some text", categories, api)
        assert result == "python_docs"

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="TUTORIAL")
        categories = ["python_docs", "tutorial", "general"]
        result = await self.categorizer.categorize("some text", categories, api)
        assert result == "tutorial"


# ── ContentDeduplicator ────────────────────────────────────────────────


class TestContentDeduplicator:
    def setup_method(self) -> None:
        self.dedup = ContentDeduplicator()

    @pytest.mark.asyncio
    async def test_no_duplicate_empty_collection(self) -> None:
        api = AsyncMock()
        api.memory_recall = AsyncMock(return_value=[])
        result = await self.dedup.is_duplicate("new text", 0.88, api)
        assert result is False

    @pytest.mark.asyncio
    async def test_duplicate_found(self) -> None:
        api = AsyncMock()
        api.memory_recall = AsyncMock(
            return_value=[{"id": "abc", "score": 0.95, "text": "similar"}]
        )
        result = await self.dedup.is_duplicate("similar text", 0.88, api)
        assert result is True

    @pytest.mark.asyncio
    async def test_below_threshold(self) -> None:
        api = AsyncMock()
        api.memory_recall = AsyncMock(
            return_value=[{"id": "abc", "score": 0.5, "text": "different"}]
        )
        result = await self.dedup.is_duplicate("different text", 0.88, api)
        assert result is False

    @pytest.mark.asyncio
    async def test_memory_error_returns_false(self) -> None:
        api = AsyncMock()
        api.memory_recall = AsyncMock(side_effect=RuntimeError("DB error"))
        result = await self.dedup.is_duplicate("text", 0.88, api)
        assert result is False


# ── QualityScorer ──────────────────────────────────────────────────────


class TestQualityScorer:
    def setup_method(self) -> None:
        self.scorer = QualityScorer()

    @pytest.mark.asyncio
    async def test_score_parsed(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="4")
        score = await self.scorer.score("good quality text", api)
        assert score == 4

    @pytest.mark.asyncio
    async def test_score_from_verbose_response(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="I would rate this a 5 out of 5")
        score = await self.scorer.score("excellent text", api)
        assert score == 5

    @pytest.mark.asyncio
    async def test_parse_failure_fallback(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="This text is medium quality")
        score = await self.scorer.score("text", api)
        assert score == 3  # Fallback

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self) -> None:
        api = AsyncMock()
        api.llm_chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        score = await self.scorer.score("text", api)
        assert score == 3  # Fallback


# ── KnowledgeCrawler ──────────────────────────────────────────────────


class TestKnowledgeCrawler:
    @pytest.mark.asyncio
    async def test_search_topic_success(self) -> None:
        mock_plugin = AsyncMock()
        mock_plugin._handle_search = AsyncMock(
            return_value={
                "query": "python async",
                "results": [
                    {"title": "Page 1", "url": "https://example.com/1", "snippet": "..."},
                    {"title": "Page 2", "url": "https://example.com/2", "snippet": "..."},
                ],
            }
        )
        crawler = KnowledgeCrawler(mock_plugin)
        results = await crawler.search_topic("python async", max_pages=5)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/1"

    @pytest.mark.asyncio
    async def test_search_topic_deduplicates_urls(self) -> None:
        mock_plugin = AsyncMock()
        mock_plugin._handle_search = AsyncMock(
            return_value={
                "query": "test",
                "results": [
                    {"title": "Page 1", "url": "https://example.com/1", "snippet": "..."},
                    {"title": "Page 1 dup", "url": "https://example.com/1", "snippet": "..."},
                    {"title": "Page 2", "url": "https://example.com/2", "snippet": "..."},
                ],
            }
        )
        crawler = KnowledgeCrawler(mock_plugin)
        results = await crawler.search_topic("test")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self) -> None:
        mock_plugin = AsyncMock()
        mock_plugin._handle_search = AsyncMock(
            return_value={"error": "timeout"}
        )
        crawler = KnowledgeCrawler(mock_plugin)
        results = await crawler.search_topic("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_page_success(self) -> None:
        mock_plugin = AsyncMock()
        mock_plugin._handle_fetch = AsyncMock(
            return_value={
                "url": "https://example.com",
                "title": "Example",
                "content": "Some content here.",
            }
        )
        crawler = KnowledgeCrawler(mock_plugin)
        result = await crawler.fetch_page("https://example.com")
        assert result is not None
        assert result["content"] == "Some content here."

    @pytest.mark.asyncio
    async def test_fetch_page_error(self) -> None:
        mock_plugin = AsyncMock()
        mock_plugin._handle_fetch = AsyncMock(
            return_value={"error": "404 Not Found"}
        )
        crawler = KnowledgeCrawler(mock_plugin)
        result = await crawler.fetch_page("https://example.com/404")
        assert result is None


# ── TrainingDataExporter ──────────────────────────────────────────────


class TestTrainingDataExporter:
    @pytest.mark.asyncio
    async def test_export_jsonl(self) -> None:
        exporter = TrainingDataExporter()
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="What is Python asyncio?")

        with tempfile.TemporaryDirectory() as tmpdir:
            async with aiosqlite.connect(":memory:") as db:
                await db.executescript(
                    """
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        quality_score INTEGER,
                        source_url TEXT,
                        source_title TEXT,
                        created_at REAL
                    );
                    INSERT INTO chunks VALUES (
                        'c1', 'Python asyncio is a library for async programming.',
                        'python_docs', 4, 'https://docs.python.org', 'Python Docs',
                        1700000000.0
                    );
                    """
                )
                await db.commit()

                result = await exporter.export_jsonl(
                    db=db,
                    export_path=tmpdir,
                    min_quality=3,
                    api=api,
                )

            assert result["exported"] == 1
            assert result["format"] == "jsonl"
            assert result["file"] is not None

            # JSONL-Datei lesen und pruefen
            filepath = Path(result["file"])
            assert filepath.exists()
            with open(filepath, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["instruction"] == "What is Python asyncio?"
            assert "asyncio" in record["output"]
            assert record["category"] == "python_docs"

    @pytest.mark.asyncio
    async def test_export_alpaca(self) -> None:
        exporter = TrainingDataExporter()
        api = AsyncMock()
        api.llm_chat = AsyncMock(return_value="Explain Python asyncio.")

        with tempfile.TemporaryDirectory() as tmpdir:
            async with aiosqlite.connect(":memory:") as db:
                await db.executescript(
                    """
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        quality_score INTEGER,
                        source_url TEXT,
                        source_title TEXT,
                        created_at REAL
                    );
                    INSERT INTO chunks VALUES (
                        'c1', 'Asyncio provides infrastructure for async IO.',
                        'python_docs', 5, 'https://docs.python.org', 'Docs',
                        1700000000.0
                    );
                    """
                )
                await db.commit()

                result = await exporter.export_alpaca(
                    db=db,
                    export_path=tmpdir,
                    min_quality=3,
                    api=api,
                )

            assert result["exported"] == 1
            assert result["format"] == "alpaca"

            filepath = Path(result["file"])
            assert filepath.exists()
            with open(filepath, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            assert len(records) == 1
            assert records[0]["instruction"] == "Explain Python asyncio."

    @pytest.mark.asyncio
    async def test_export_empty(self) -> None:
        exporter = TrainingDataExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            async with aiosqlite.connect(":memory:") as db:
                await db.executescript(
                    """
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        quality_score INTEGER,
                        source_url TEXT,
                        source_title TEXT,
                        created_at REAL
                    );
                    """
                )
                await db.commit()

                result = await exporter.export_jsonl(
                    db=db,
                    export_path=tmpdir,
                    min_quality=3,
                )

            assert result["exported"] == 0
            assert result["file"] is None

    @pytest.mark.asyncio
    async def test_export_without_api_fallback_instruction(self) -> None:
        exporter = TrainingDataExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            async with aiosqlite.connect(":memory:") as db:
                await db.executescript(
                    """
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        quality_score INTEGER,
                        source_url TEXT,
                        source_title TEXT,
                        created_at REAL
                    );
                    INSERT INTO chunks VALUES (
                        'c1', 'Some content about programming.',
                        'general', 4, 'https://example.com', 'Example',
                        1700000000.0
                    );
                    """
                )
                await db.commit()

                # Kein API-Objekt: Fallback-Instruktion
                result = await exporter.export_jsonl(
                    db=db,
                    export_path=tmpdir,
                    min_quality=3,
                    api=None,
                )

            assert result["exported"] == 1
            filepath = Path(result["file"])
            with open(filepath, "r", encoding="utf-8") as fh:
                record = json.loads(fh.readline())
            assert record["instruction"].startswith("Explain the following:")
