"""
Lexy AI - Orchestrator Task Queue.

Priority-based task queue with SQLite persistence. Tasks flow through
the lifecycle: ``queued`` -> ``running`` -> ``done`` / ``failed``.

Failed tasks with ``retry_count < 2`` can be requeued automatically
by the orchestrator. Priorities: low (0), normal (1), high (2).

Used by ``OrchestratorPlugin`` to manage delegated work items.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import aiosqlite

from lexy_core.utils.logging import get_logger

log = get_logger(module="task_queue")

PRIORITY_MAP: dict[str, int] = {"low": 0, "normal": 1, "high": 2}


@dataclass
class QueuedTask:
    """In-memory representation of a queued task (for type hints / docs)."""

    id: str
    task: str
    persona: str | None
    brain: str
    priority: int
    status: str
    agent_id: str | None
    result_summary: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    retry_count: int
    schedule_label: str | None = None


class TaskQueue:
    """
    SQLite-backed priority task queue for the orchestrator.

    All state is persisted; the queue survives restarts.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def init_tables(self) -> None:
        """Create the tasks table if it does not exist."""
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id              TEXT PRIMARY KEY,
                task            TEXT NOT NULL,
                persona         TEXT,
                brain           TEXT NOT NULL DEFAULT 'e4b',
                priority        INTEGER NOT NULL DEFAULT 1,
                status          TEXT NOT NULL DEFAULT 'queued',
                agent_id        TEXT,
                result_summary  TEXT,
                created_at      REAL NOT NULL,
                started_at      REAL,
                finished_at     REAL,
                retry_count     INTEGER DEFAULT 0,
                schedule_label  TEXT
            )"""
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status "
            "ON tasks(status, priority DESC, created_at ASC)"
        )
        await self._db.commit()
        log.info("task_queue.tables_ready")

    async def enqueue(
        self,
        task: str,
        persona: str | None = None,
        brain: str = "e4b",
        priority: str | int = "normal",
        schedule_label: str | None = None,
    ) -> str:
        """
        Add a new task to the queue.

        Args:
            task:           The task description.
            persona:        Optional persona ID for the agent.
            brain:          LLM brain to use.
            priority:       ``"low"`` / ``"normal"`` / ``"high"`` or 0/1/2.
            schedule_label: Scheduler label (if triggered by scheduler).

        Returns:
            The generated task ID.
        """
        task_id = uuid.uuid4().hex[:12]
        if isinstance(priority, str):
            prio = PRIORITY_MAP.get(priority, 1)
        else:
            prio = priority

        await self._db.execute(
            "INSERT INTO tasks "
            "(id, task, persona, brain, priority, status, created_at, schedule_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, task, persona, brain, prio, "queued", time.time(), schedule_label),
        )
        await self._db.commit()

        log.info(
            "task_queue.enqueued",
            task_id=task_id,
            persona=persona,
            priority=prio,
        )
        return task_id

    async def dequeue(self) -> dict[str, Any] | None:
        """
        Get the highest-priority queued task without marking it running.

        Returns:
            Task dict or None if the queue is empty.
        """
        async with self._db.execute(
            "SELECT id, task, persona, brain, priority, retry_count, schedule_label "
            "FROM tasks WHERE status = 'queued' "
            "ORDER BY priority DESC, created_at ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "task": row[1],
                "persona": row[2],
                "brain": row[3],
                "priority": row[4],
                "retry_count": row[5],
                "schedule_label": row[6],
            }

    async def mark_running(self, task_id: str, agent_id: str) -> None:
        """Mark a task as running with the given agent_id."""
        await self._db.execute(
            "UPDATE tasks SET status = 'running', agent_id = ?, started_at = ? "
            "WHERE id = ?",
            (agent_id, time.time(), task_id),
        )
        await self._db.commit()
        log.debug("task_queue.mark_running", task_id=task_id, agent_id=agent_id)

    async def mark_done(self, task_id: str, result: str) -> None:
        """Mark a task as successfully completed."""
        await self._db.execute(
            "UPDATE tasks SET status = 'done', result_summary = ?, finished_at = ? "
            "WHERE id = ?",
            (result[:500], time.time(), task_id),
        )
        await self._db.commit()
        log.info("task_queue.mark_done", task_id=task_id)

    async def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed, incrementing retry_count."""
        await self._db.execute(
            "UPDATE tasks SET status = 'failed', result_summary = ?, "
            "finished_at = ?, retry_count = retry_count + 1 WHERE id = ?",
            (error[:500], time.time(), task_id),
        )
        await self._db.commit()
        log.warning("task_queue.mark_failed", task_id=task_id, error=error[:200])

    async def get_retryable(self) -> list[dict[str, Any]]:
        """Get failed tasks with retry_count < 2 (eligible for retry)."""
        async with self._db.execute(
            "SELECT id, task, persona, brain, retry_count "
            "FROM tasks WHERE status = 'failed' AND retry_count < 2 "
            "ORDER BY priority DESC"
        ) as cur:
            return [
                {
                    "id": r[0],
                    "task": r[1],
                    "persona": r[2],
                    "brain": r[3],
                    "retry_count": r[4],
                }
                async for r in cur
            ]

    async def requeue(self, task_id: str) -> None:
        """Move a task back to queued status for retry."""
        await self._db.execute(
            "UPDATE tasks SET status = 'queued', agent_id = NULL, "
            "started_at = NULL, finished_at = NULL WHERE id = ?",
            (task_id,),
        )
        await self._db.commit()
        log.info("task_queue.requeued", task_id=task_id)

    async def find_by_agent(self, agent_id: str) -> dict[str, Any] | None:
        """Find a running task by its agent_id."""
        async with self._db.execute(
            "SELECT id, task, persona, brain, priority, retry_count, schedule_label "
            "FROM tasks WHERE agent_id = ? AND status = 'running' LIMIT 1",
            (agent_id,),
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "task": row[1],
                "persona": row[2],
                "brain": row[3],
                "priority": row[4],
                "retry_count": row[5],
                "schedule_label": row[6],
            }

    async def find_by_schedule_label(self, label: str) -> dict[str, Any] | None:
        """Find the most recent task with a given schedule_label."""
        async with self._db.execute(
            "SELECT id, task, persona, brain, priority, status "
            "FROM tasks WHERE schedule_label = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (label,),
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "task": row[1],
                "persona": row[2],
                "brain": row[3],
                "priority": row[4],
                "status": row[5],
            }

    async def list_pending(self) -> list[dict[str, Any]]:
        """List all queued and running tasks."""
        async with self._db.execute(
            "SELECT id, task, persona, status, priority, created_at "
            "FROM tasks WHERE status IN ('queued', 'running') "
            "ORDER BY priority DESC, created_at ASC"
        ) as cur:
            return [
                {
                    "id": r[0],
                    "task": r[1],
                    "persona": r[2],
                    "status": r[3],
                    "priority": r[4],
                    "created_at": r[5],
                }
                async for r in cur
            ]

    async def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """List the most recently created tasks (any status)."""
        async with self._db.execute(
            "SELECT id, task, persona, status, result_summary, created_at, finished_at "
            "FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [
                {
                    "id": r[0],
                    "task": r[1],
                    "persona": r[2],
                    "status": r[3],
                    "result_summary": r[4],
                    "created_at": r[5],
                    "finished_at": r[6],
                }
                async for r in cur
            ]
