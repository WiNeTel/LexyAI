"""Tests for the Orchestrator plugin — persona registry, task queue, brain, pool."""
from __future__ import annotations

import time
import pytest
import aiosqlite

from plugins.orchestrator.persona_registry import Persona, PersonaRegistry
from plugins.orchestrator.task_queue import TaskQueue, PRIORITY_MAP, QueuedTask
from plugins.orchestrator.orchestrator_brain import OrchestratorBrain
from plugins.orchestrator.agent_pool import OrchestratorPool


# ─── Persona ────────────────────────────────────────────────────────────────


class TestPersona:
    def test_defaults(self) -> None:
        p = Persona(id="test", name="Test", system_prompt="Hello")
        assert p.brain == "e4b"
        assert p.temperature == 0.6
        assert p.builtin is True

    def test_to_dict(self) -> None:
        p = Persona(id="x", name="X", system_prompt="Short prompt")
        d = p.to_dict()
        assert d["id"] == "x"
        assert d["name"] == "X"
        assert "brain" in d

    def test_long_prompt_truncated_in_dict(self) -> None:
        p = Persona(id="x", name="X", system_prompt="A" * 500)
        d = p.to_dict()
        assert len(d["system_prompt"]) < 500


class TestPersonaRegistry:
    def test_init_empty(self) -> None:
        reg = PersonaRegistry()
        assert reg.list_all() == []

    def test_get_nonexistent(self) -> None:
        reg = PersonaRegistry()
        assert reg.get("nope") is None

    @pytest.mark.asyncio
    async def test_init_with_config(self, tmp_path) -> None:
        db = await aiosqlite.connect(":memory:")
        reg = PersonaRegistry()
        await reg.init_db(db, {
            "tutor": {"name": "Tutor", "prompt": "Sei geduldig", "brain": "a4b", "temperature": 0.5},
            "critic": {"name": "Kritiker", "prompt": "Sei kritisch"},
        })
        assert reg.get("tutor") is not None
        assert reg.get("tutor").brain == "a4b"
        assert reg.get("critic") is not None
        assert len(reg.list_all()) == 2
        await db.close()

    @pytest.mark.asyncio
    async def test_register_and_delete(self) -> None:
        db = await aiosqlite.connect(":memory:")
        reg = PersonaRegistry()
        await reg.init_db(db, {})
        persona = await reg.register("custom", "Custom Agent", "Do stuff")
        assert persona.builtin is False
        assert reg.get("custom") is not None
        assert await reg.delete("custom") is True
        assert reg.get("custom") is None
        await db.close()

    @pytest.mark.asyncio
    async def test_cannot_delete_builtin(self) -> None:
        db = await aiosqlite.connect(":memory:")
        reg = PersonaRegistry()
        await reg.init_db(db, {"builtin1": {"name": "B", "prompt": "P"}})
        assert await reg.delete("builtin1") is False
        assert reg.get("builtin1") is not None
        await db.close()


# ─── TaskQueue ──────────────────────────────────────────────────────────────


class TestPriorityMap:
    def test_values(self) -> None:
        assert PRIORITY_MAP["low"] == 0
        assert PRIORITY_MAP["normal"] == 1
        assert PRIORITY_MAP["high"] == 2


class TestTaskQueue:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self) -> None:
        db = await aiosqlite.connect(":memory:")
        q = TaskQueue(db)
        await q.init_tables()
        task_id = await q.enqueue("Do something", persona="researcher")
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        item = await q.dequeue()
        assert item is not None
        assert item["task"] == "Do something"
        assert item["persona"] == "researcher"
        await db.close()

    @pytest.mark.asyncio
    async def test_dequeue_empty(self) -> None:
        db = await aiosqlite.connect(":memory:")
        q = TaskQueue(db)
        await q.init_tables()
        assert await q.dequeue() is None
        await db.close()

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        db = await aiosqlite.connect(":memory:")
        q = TaskQueue(db)
        await q.init_tables()
        await q.enqueue("Low task", priority="low")
        await q.enqueue("High task", priority="high")
        await q.enqueue("Normal task", priority="normal")
        item = await q.dequeue()
        assert item["task"] == "High task"
        await db.close()

    @pytest.mark.asyncio
    async def test_mark_done(self) -> None:
        db = await aiosqlite.connect(":memory:")
        q = TaskQueue(db)
        await q.init_tables()
        tid = await q.enqueue("Task 1")
        await q.mark_running(tid, "agent-123")
        await q.mark_done(tid, "Result OK")
        pending = await q.list_pending()
        assert len(pending) == 0
        recent = await q.list_recent(1)
        assert len(recent) == 1
        assert recent[0]["status"] == "done"
        await db.close()

    @pytest.mark.asyncio
    async def test_mark_failed_and_retryable(self) -> None:
        db = await aiosqlite.connect(":memory:")
        q = TaskQueue(db)
        await q.init_tables()
        tid = await q.enqueue("Failing task")
        await q.mark_running(tid, "a1")
        await q.mark_failed(tid, "Error occurred")
        retryable = await q.get_retryable()
        assert len(retryable) == 1
        assert retryable[0]["id"] == tid
        await db.close()

    @pytest.mark.asyncio
    async def test_schedule_label(self) -> None:
        db = await aiosqlite.connect(":memory:")
        q = TaskQueue(db)
        await q.init_tables()
        tid = await q.enqueue("Daily news", schedule_label="orch:daily_news")
        item = await q.dequeue()
        assert item["schedule_label"] == "orch:daily_news"
        await db.close()


# ─── OrchestratorBrain ──────────────────────────────────────────────────────


class TestOrchestratorBrain:
    def test_init(self) -> None:
        from unittest.mock import MagicMock
        api = MagicMock()
        brain = OrchestratorBrain(api, brain="e4b", max_tokens=200)
        assert brain._brain == "e4b"
        assert brain._max_tokens == 200

    def test_decision_log_empty(self) -> None:
        from unittest.mock import MagicMock
        brain = OrchestratorBrain(MagicMock())
        assert brain._decision_log == []


# ─── OrchestratorPool ──────────────────────────────────────────────────────


class TestOrchestratorPool:
    def test_list_running_empty(self) -> None:
        from unittest.mock import MagicMock
        manager = MagicMock()
        manager.list_agents.return_value = []
        registry = MagicMock()
        pool = OrchestratorPool(api=MagicMock(), manager=manager, registry=registry)
        assert pool.list_running() == []

    def test_get_nonexistent(self) -> None:
        from unittest.mock import MagicMock
        manager = MagicMock()
        manager.get_agent.return_value = None
        pool = OrchestratorPool(api=MagicMock(), manager=manager, registry=MagicMock())
        assert pool.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_spawn_unknown_persona(self) -> None:
        from unittest.mock import MagicMock
        manager = MagicMock()
        registry = PersonaRegistry()
        pool = OrchestratorPool(api=MagicMock(), manager=manager, registry=registry)
        result = await pool.spawn_with_persona("task", "unknown_persona")
        assert "error" in result
