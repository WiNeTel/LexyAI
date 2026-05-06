"""
Error-learning helper — one-call interface to "what similar mistakes
have we made before?".

The coder_brain plugin records every failed subprocess run + every
exception caught during a coding step into ChromaDB's ``errors``
collection (via the ``memory_store`` API). Before the brain plans the
next attempt, it asks this helper for the top-N similar past errors
plus their successful resolutions (looked up from the ``solutions``
collection by the same task-tag), and feeds them into the prompt as
"Vorherige Lehren".

This is the cheapest possible learning loop — no fine-tuning, no
re-training. The LLM does the matching itself once we surface the
right snippets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any


log = logging.getLogger(__name__)


@dataclass
class ErrorMemory:
    """A past error + (optional) solution as seen by the brain."""

    text: str
    metadata: dict[str, Any]
    solution: str = ""           # optional follow-up from the solutions collection
    distance: float = 0.0        # 0 = exact, higher = less similar


class ErrorLearning:
    """Tiny façade over MemoryManager for error-recall + success-store."""

    def __init__(self, memory: Any, *, recall_limit: int = 3) -> None:
        self._memory = memory
        self._recall_limit = max(1, int(recall_limit))

    # ─── Recall ──────────────────────────────────────────────────────

    async def recall_similar(
        self,
        *,
        query: str,
        task_tag: str = "",
    ) -> list[ErrorMemory]:
        """Find similar past errors. Best-effort, returns ``[]`` on failure.

        ``task_tag`` is an optional discriminator ("self_coder/skill/foo")
        so memory from a totally unrelated project doesn't leak in.
        """
        if self._memory is None or not (query or "").strip():
            return []
        try:
            hits = await self._memory.recall(
                query=query,
                collection="errors",
                limit=self._recall_limit,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("coder.error_recall_failed err=%s", exc)
            return []
        out: list[ErrorMemory] = []
        for hit in hits or []:
            md = hit.get("metadata") or {}
            if task_tag and md.get("task_tag") and md.get("task_tag") != task_tag:
                # Different task — skip cross-pollination unless we have
                # nothing else. (Caller decides if it's better than nothing.)
                continue
            out.append(
                ErrorMemory(
                    text=str(hit.get("content") or hit.get("text") or ""),
                    metadata=md,
                    distance=float(hit.get("distance") or 0.0),
                )
            )
        # Best-effort: try to find a "solution" record stored against the
        # same task_tag. We don't fail the whole recall if this lookup
        # bombs.
        if task_tag:
            try:
                sol_hits = await self._memory.recall(
                    query=query,
                    collection="solutions",
                    limit=2,
                    metadata_equals={"task_tag": task_tag},
                )
                if sol_hits:
                    sol_text = "\n".join(
                        str(h.get("content") or "") for h in sol_hits[:2]
                    )
                    if out:
                        out[0].solution = sol_text
            except Exception:  # noqa: BLE001
                pass
        return out

    # ─── Store ──────────────────────────────────────────────────────

    async def remember_failure(
        self,
        *,
        text: str,
        task_tag: str = "",
        extras: dict[str, Any] | None = None,
    ) -> bool:
        """Persist an error as ``errors`` collection memory."""
        if self._memory is None or not (text or "").strip():
            return False
        meta: dict[str, Any] = {
            "source": "coder",
            "kind": "code_runtime",
            "task_tag": task_tag,
            "stored_at": time.time(),
        }
        if extras:
            meta.update(extras)
        try:
            await self._memory.store(
                text=text[:4000],
                collection="errors",
                metadata=meta,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("coder.error_store_failed err=%s", exc)
            return False

    async def remember_solution(
        self,
        *,
        text: str,
        task_tag: str = "",
        extras: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a successful resolution to the ``solutions`` collection."""
        if self._memory is None or not (text or "").strip():
            return False
        meta: dict[str, Any] = {
            "source": "coder",
            "kind": "code_solution",
            "task_tag": task_tag,
            "stored_at": time.time(),
        }
        if extras:
            meta.update(extras)
        try:
            await self._memory.store(
                text=text[:4000],
                collection="solutions",
                metadata=meta,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("coder.solution_store_failed err=%s", exc)
            return False
