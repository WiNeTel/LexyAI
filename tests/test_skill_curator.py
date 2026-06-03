"""
Tests for the Phase-P3 skill curator.

Two layers:

1. ``plan_transitions`` — the pure lifecycle decision function. Driven with
   stand-in entries (no DB, no filesystem): stale/archive ageing,
   reactivation, the low-success safety net, and the pinned / managed-source
   exemptions.
2. ``SkillCurator`` archive → restore — a real aiosqlite registry + tmp skill
   folders, asserting the folder actually moves into ``.archive/`` and back
   and that ``state`` / ``status`` track it (recoverable, never deleted).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiosqlite
import pytest

from plugins.skill_writer.skill_curator import (
    SkillCurator,
    plan_transitions,
)
from plugins.skill_writer.skill_registry import SkillRegistry

_NOW = 1_000_000_000.0
_DAY = 86400.0

_THRESHOLDS: dict[str, Any] = {
    "stale_days": 30,
    "archive_days": 90,
    "min_success_rate": 0.4,
    "min_runs": 5,
}


def _entry(
    name: str,
    *,
    source: str = "auto_pattern",
    state: str = "active",
    pinned: bool = False,
    used_days_ago: float | None = None,
    usage_count: int = 0,
    success_count: int = 0,
) -> SimpleNamespace:
    last = None if used_days_ago is None else _NOW - used_days_ago * _DAY
    return SimpleNamespace(
        name=name,
        source=source,
        state=state,
        pinned=pinned,
        last_used_at=last,
        created_at=_NOW - 365 * _DAY,  # born long ago
        usage_count=usage_count,
        success_count=success_count,
    )


def _plan(entries: list[SimpleNamespace]) -> dict[str, str]:
    """Return {name: to_state} for the planned transitions."""
    return {
        t.name: t.to_state
        for t in plan_transitions(entries, now=_NOW, **_THRESHOLDS)
    }


# ─── Pure transition logic ──────────────────────────────────────────────────


def test_active_to_stale() -> None:
    plan = _plan([_entry("s", state="active", used_days_ago=40)])
    assert plan == {"s": "stale"}


def test_stale_to_archived() -> None:
    plan = _plan([_entry("s", state="stale", used_days_ago=100)])
    assert plan == {"s": "archived"}


def test_active_straight_to_archived_when_very_old() -> None:
    plan = _plan([_entry("s", state="active", used_days_ago=120)])
    assert plan == {"s": "archived"}


def test_recent_use_reactivates_stale() -> None:
    plan = _plan([_entry("s", state="stale", used_days_ago=1)])
    assert plan == {"s": "active"}


def test_fresh_active_has_no_transition() -> None:
    assert _plan([_entry("s", state="active", used_days_ago=2)]) == {}


def test_pinned_is_exempt() -> None:
    assert _plan([_entry("s", state="active", pinned=True, used_days_ago=200)]) == {}


def test_manual_source_is_never_touched() -> None:
    assert _plan([_entry("s", source="manual", used_days_ago=200)]) == {}
    assert _plan([_entry("s", source="imported", used_days_ago=200)]) == {}


def test_low_success_auto_skill_is_archived() -> None:
    # rate 0.2 < 0.4 after >= 5 runs, even though recently used.
    plan = _plan(
        [_entry("s", used_days_ago=1, usage_count=10, success_count=2)]
    )
    assert plan == {"s": "archived"}


def test_low_success_below_min_runs_is_ignored() -> None:
    # Only 3 runs → not enough evidence; recent → no age transition either.
    plan = _plan(
        [_entry("s", used_days_ago=1, usage_count=3, success_count=0)]
    )
    assert plan == {}


def test_high_success_recent_skill_survives() -> None:
    plan = _plan(
        [_entry("s", used_days_ago=1, usage_count=10, success_count=9)]
    )
    assert plan == {}


# ─── Registry-backed archive / restore round-trip ───────────────────────────


class _FakeAPI:
    async def ws_broadcast(self, payload: dict[str, Any]) -> None:
        pass


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_skill_folder(root: Path, name: str) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo skill\n---\nbody text\n",
        encoding="utf-8",
    )
    return folder


@pytest.fixture()
def registry(tmp_path: Path):
    skills_path = tmp_path / "skills"
    skills_path.mkdir()
    db = _run(aiosqlite.connect(":memory:"))
    reg = SkillRegistry(db=db, skills_path=skills_path)
    _run(reg.init_tables())
    yield reg, skills_path
    _run(db.close())


def test_curator_archive_then_restore_round_trip(registry: Any) -> None:
    reg, skills_path = registry
    folder = _make_skill_folder(skills_path, "demo")
    _run(
        reg.register(
            name="demo",
            description="demo skill",
            file_path=str(folder),
            source="auto_pattern",
        )
    )
    curator = SkillCurator(registry=reg, skills_path=skills_path, api=_FakeAPI())

    # Archive — folder moves to .archive/, state/status flip, recoverable.
    _run(curator._archive_skill("demo"))  # noqa: SLF001
    entry = _run(reg.get("demo"))
    assert entry.state == "archived"
    assert entry.status == "disabled"
    assert not folder.exists()
    assert ".archive" in entry.file_path
    assert Path(entry.file_path).is_dir()

    # Restore — folder comes back, state/status reactivated.
    assert _run(curator.restore("demo")) is True
    entry2 = _run(reg.get("demo"))
    assert entry2.state == "active"
    assert entry2.status == "active"
    assert (skills_path / "demo").is_dir()


def test_curator_run_dry_run_does_not_mutate(registry: Any) -> None:
    reg, skills_path = registry
    folder = _make_skill_folder(skills_path, "old")
    _run(
        reg.register(
            name="old",
            description="old skill",
            file_path=str(folder),
            source="auto_pattern",
        )
    )
    # Age it well past archive_days by rewriting created_at directly.
    _run(
        reg._db.execute(  # noqa: SLF001
            "UPDATE skills SET created_at = ? WHERE name = ?",
            (_NOW - 200 * _DAY, "old"),
        )
    )
    _run(reg._db.commit())  # noqa: SLF001
    curator = SkillCurator(registry=reg, skills_path=skills_path, api=_FakeAPI())

    report = _run(curator.run(dry_run=True))
    assert report["dry_run"] is True
    assert report["count"] == 1
    # Nothing actually changed.
    entry = _run(reg.get("old"))
    assert entry.state == "active"
    assert folder.is_dir()

    # Real run applies it.
    report2 = _run(curator.run(dry_run=False))
    assert report2["count"] == 1
    entry2 = _run(reg.get("old"))
    assert entry2.state == "archived"


def test_curator_pin_blocks_archival(registry: Any) -> None:
    reg, skills_path = registry
    _make_skill_folder(skills_path, "keep")
    _run(
        reg.register(
            name="keep",
            description="keep skill",
            file_path=str(skills_path / "keep"),
            source="auto_pattern",
        )
    )
    _run(
        reg._db.execute(  # noqa: SLF001
            "UPDATE skills SET created_at = ? WHERE name = ?",
            (_NOW - 200 * _DAY, "keep"),
        )
    )
    _run(reg._db.commit())  # noqa: SLF001
    curator = SkillCurator(registry=reg, skills_path=skills_path, api=_FakeAPI())

    _run(curator.set_pinned("keep", True))
    report = _run(curator.run(dry_run=False))
    assert report["count"] == 0  # pinned → no transition
    entry = _run(reg.get("keep"))
    assert entry.state == "active"
