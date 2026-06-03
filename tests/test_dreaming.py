"""
Tests for the Phase-P1 dreaming upgrades:

* ``_op_deduplicate`` now **archives** the merged originals (recoverable)
  instead of only logging their ids — the old "future maintenance job"
  TODO, finished.
* ``_op_decay`` archives entries older than ``decay_days`` that were never
  recalled (``access_count == 0``), keeps fresh / used / consolidated ones,
  and honours ``decay_dry_run``.
* a ``maintenance_only`` cycle runs decay without the LLM-driven ops.

Driven through a small fake PluginAPI — no LexyApp, no services, no LLM.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from plugins.dreaming.dreaming_plugin import DreamingPlugin

_DAY = 86400


class _FakeAPI:
    """Stand-in PluginAPI capturing archive / store / emit calls."""

    def __init__(
        self,
        *,
        recall_queue: list[list[dict[str, Any]]] | None = None,
        browse_pages: dict[str, list[tuple[list[dict[str, Any]], int]]] | None = None,
        llm_response: str = "Ein zusammengefuehrter Eintrag mit Inhalt.",
    ) -> None:
        self._cfg: dict[str, Any] = {
            "enabled": True,
            "interval_minutes": 120,
            "quiet_hours": ["02:00", "06:00"],
            "max_operations_per_cycle": 10,
            "similarity_threshold": 0.85,
            "decay_days": 90,
            "min_idle_minutes": 30,
            "archive_merged_originals": True,
            "decay_enabled": True,
            "decay_requires_zero_access": True,
            "decay_dry_run": False,
            "decay_collections": ["context"],
            "maintenance_ignores_quiet_hours": True,
        }
        self._recall_queue = list(recall_queue or [])
        self._browse_pages = dict(browse_pages or {})
        self._llm_response = llm_response

        self.archived: list[dict[str, Any]] = []
        self.stored: list[dict[str, Any]] = []
        self.emitted: list[tuple[str, dict[str, Any]]] = []
        self.broadcasts: list[dict[str, Any]] = []

    def get_config(self) -> dict[str, Any]:
        return dict(self._cfg)

    async def memory_recall(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self._recall_queue:
            return self._recall_queue.pop(0)
        return []

    async def llm_chat(self, **kwargs: Any) -> str:
        return self._llm_response

    async def memory_store(
        self,
        text: str,
        collection: str = "facts",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.stored.append(
            {"text": text, "collection": collection, "metadata": dict(metadata or {})}
        )
        return f"stored-{len(self.stored)}"

    async def memory_archive(
        self,
        collection: str,
        ids: list[str],
        reason: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        self.archived.append(
            {
                "collection": collection,
                "ids": list(ids),
                "reason": reason,
                "extra_meta": dict(extra_meta or {}),
            }
        )
        return {"archived": len(ids), "fts": len(ids)}

    async def memory_browse(
        self, collection: str = "facts", page: int = 1, limit: int = 200
    ) -> tuple[list[dict[str, Any]], int]:
        pages = self._browse_pages.get(collection, [])
        idx = page - 1
        if 0 <= idx < len(pages):
            return pages[idx]
        return [], 0

    async def emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        self.emitted.append((event, dict(data or {})))

    async def ws_broadcast(self, payload: dict[str, Any]) -> None:
        self.broadcasts.append(dict(payload))


def _plugin(api: _FakeAPI) -> DreamingPlugin:
    return DreamingPlugin(api, SimpleNamespace(name="dreaming"))


# ─── Dedup now archives ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_archives_both_originals() -> None:
    seed = {"id": "seed1", "content": "Mike wohnt in Berlin", "score": 1.0}
    dupe = {"id": "dupe1", "content": "Mikes Zuhause ist Berlin", "score": 0.93}
    api = _FakeAPI(recall_queue=[[seed], [seed, dupe]])
    plugin = _plugin(api)
    await plugin.on_load()

    result = await plugin._op_deduplicate()

    assert result is not None
    assert result["op"] == "deduplicate"
    assert len(api.stored) == 1  # merged entry written
    assert len(api.archived) == 1  # originals archived, not logged
    call = api.archived[0]
    assert set(call["ids"]) == {"seed1", "dupe1"}
    assert call["reason"] == "dedup_merge"
    assert call["extra_meta"]["merged_into"] == result["merged_id"]
    assert result["archived"] == 2


@pytest.mark.asyncio
async def test_dedup_keeps_originals_when_archive_disabled() -> None:
    seed = {"id": "seed1", "content": "Mike wohnt in Berlin", "score": 1.0}
    dupe = {"id": "dupe1", "content": "Mikes Zuhause ist Berlin", "score": 0.93}
    api = _FakeAPI(recall_queue=[[seed], [seed, dupe]])
    plugin = _plugin(api)
    await plugin.on_load()
    plugin._archive_merged_originals = False  # noqa: SLF001

    result = await plugin._op_deduplicate()

    assert result is not None
    assert api.archived == []
    assert result["archived"] == 0


# ─── Decay archives only old + unused ───────────────────────────────────────


def _decay_items() -> list[dict[str, Any]]:
    now = time.time()
    old = now - 120 * _DAY
    return [
        {"id": "old_unused", "content": "alt+ungenutzt", "metadata": {"created_at": old, "access_count": 0}},
        {"id": "old_used", "content": "alt aber genutzt", "metadata": {"created_at": old, "access_count": 7}},
        {"id": "fresh", "content": "frisch", "metadata": {"created_at": now, "access_count": 0}},
        {"id": "consolidated", "content": "merged", "metadata": {"created_at": old, "access_count": 0, "type": "merged"}},
    ]


@pytest.mark.asyncio
async def test_decay_archives_old_unused_only() -> None:
    items = _decay_items()
    api = _FakeAPI(browse_pages={"context": [(items, len(items))]})
    plugin = _plugin(api)
    await plugin.on_load()

    result = await plugin._op_decay()

    assert result is not None
    assert result["op"] == "decay"
    assert result["candidates"] == 1
    assert result["archived"] == 1
    assert len(api.archived) == 1
    assert api.archived[0]["ids"] == ["old_unused"]
    assert api.archived[0]["reason"] == "decay_unused"


@pytest.mark.asyncio
async def test_decay_dry_run_reports_without_archiving() -> None:
    items = _decay_items()
    api = _FakeAPI(browse_pages={"context": [(items, len(items))]})
    plugin = _plugin(api)
    await plugin.on_load()
    plugin._decay_dry_run = True  # noqa: SLF001

    result = await plugin._op_decay()

    assert result is not None
    assert result["dry_run"] is True
    assert result["candidates"] == 1
    assert result["archived"] == 0
    assert api.archived == []  # nothing touched


@pytest.mark.asyncio
async def test_decay_respects_zero_access_toggle() -> None:
    """With the zero-access guard off, age alone qualifies an item."""
    items = _decay_items()
    api = _FakeAPI(browse_pages={"context": [(items, len(items))]})
    plugin = _plugin(api)
    await plugin.on_load()
    plugin._decay_requires_zero_access = False  # noqa: SLF001

    result = await plugin._op_decay()

    assert result is not None
    # old_unused + old_used both qualify (consolidated is still kept).
    assert set(api.archived[0]["ids"]) == {"old_unused", "old_used"}


# ─── Maintenance-only cycle ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_maintenance_only_cycle_runs_decay_not_consolidation() -> None:
    items = _decay_items()
    api = _FakeAPI(browse_pages={"context": [(items, len(items))]})
    plugin = _plugin(api)
    await plugin.on_load()

    results = await plugin._run_cycle(force=True, maintenance_only=True)

    assert results, "maintenance cycle should report the decay op"
    assert all(r["op"] == "decay" for r in results)  # no dedup/link/staleness
    assert any(
        event == "core.dreaming_cycle" and data.get("maintenance_only") is True
        for event, data in api.emitted
    )
