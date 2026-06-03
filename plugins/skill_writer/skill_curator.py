"""
Lexy AI — Skill Curator (Phase P3).

Background lifecycle manager for *agent-created* skills. Hermes ships a
"Curator" that grades, ages out, and prunes self-made skills; this is Lexy's
recoverable take on it.

Two safety nets, both **recoverable** (archived skills move to
``data/skills/.archive/`` and can be restored, never hard-deleted):

* **Age-based state machine** — ``active → (stale_days unused) → stale →
  (archive_days unused) → archived``. Using a skill resets it to ``active``.
* **Low-success guard** — an auto-skill whose success rate drops below
  ``min_success_rate`` after at least ``min_runs`` runs is archived. This is
  the net that keeps fully-autonomous skill learning from accumulating junk.

Only skills whose ``source`` is in ``managed_sources`` are ever touched —
hand-written (``manual``) and ``imported`` skills are left alone — and
``pinned`` skills are exempt from everything.

The transition planner is a pure function so it can be unit-tested without a
database or the filesystem; the :class:`SkillCurator` applies the plan.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lexy_core.utils.logging import get_logger

log = get_logger(module="skill_curator")


# Sources the curator is allowed to manage. Manual / imported skills are the
# user's, never auto-archived.
DEFAULT_MANAGED_SOURCES: frozenset[str] = frozenset(
    {"auto", "auto_pattern", "self_refine", "sub_agent"}
)


@dataclass(frozen=True)
class CuratorTransition:
    """One planned lifecycle change for a skill."""

    name: str
    from_state: str
    to_state: str
    reason: str

    def to_public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "from": self.from_state,
            "to": self.to_state,
            "reason": self.reason,
        }


def plan_transitions(
    entries: Iterable[Any],
    *,
    now: float,
    stale_days: int,
    archive_days: int,
    min_success_rate: float,
    min_runs: int,
    managed_sources: Iterable[str] = DEFAULT_MANAGED_SOURCES,
) -> list[CuratorTransition]:
    """Compute curator transitions for a set of skill entries (pure).

    ``entries`` are objects exposing ``name``, ``source``, ``state``,
    ``pinned``, ``last_used_at``, ``created_at``, ``usage_count`` and
    ``success_count`` (a :class:`SkillEntry` fits, as does any stand-in).
    """
    managed = set(managed_sources)
    plans: list[CuratorTransition] = []
    for entry in entries:
        if getattr(entry, "pinned", False):
            continue
        if entry.source not in managed:
            continue

        state = entry.state or "active"
        usage = int(entry.usage_count or 0)
        success = int(entry.success_count or 0)

        # 1) Low-success guard (only meaningful once we have enough runs).
        if usage >= min_runs:
            rate = success / usage if usage else 1.0
            if rate < min_success_rate:
                if state != "archived":
                    plans.append(
                        CuratorTransition(
                            entry.name, state, "archived", f"low_success:{rate:.2f}"
                        )
                    )
                continue  # archived → nothing else to decide

        # 2) Age-based state machine on idle time.
        last = entry.last_used_at or entry.created_at or now
        idle_days = max(0.0, (now - float(last)) / 86400.0)

        if idle_days >= archive_days:
            if state != "archived":
                plans.append(
                    CuratorTransition(
                        entry.name, state, "archived", f"unused_{int(idle_days)}d"
                    )
                )
        elif idle_days >= stale_days:
            if state == "active":
                plans.append(
                    CuratorTransition(
                        entry.name, "active", "stale", f"unused_{int(idle_days)}d"
                    )
                )
        else:
            # Recently used → bring a stale skill back to life.
            if state == "stale":
                plans.append(
                    CuratorTransition(entry.name, "stale", "active", "used_again")
                )
    return plans


class SkillCurator:
    """Applies the curator plan + owns the background loop and controls."""

    def __init__(
        self,
        *,
        registry: Any,
        skills_path: Path,
        api: Any,
        stale_days: int = 30,
        archive_days: int = 90,
        min_success_rate: float = 0.4,
        min_runs: int = 5,
        interval_hours: float = 24.0,
        min_idle_minutes: float = 10.0,
        managed_sources: Iterable[str] = DEFAULT_MANAGED_SOURCES,
    ) -> None:
        self._registry = registry
        self._skills_path = Path(skills_path)
        self._api = api
        self._stale_days = stale_days
        self._archive_days = archive_days
        self._min_success_rate = min_success_rate
        self._min_runs = min_runs
        self._interval_hours = interval_hours
        self._min_idle_minutes = min_idle_minutes
        self._managed_sources = frozenset(managed_sources)

        self._last_activity: float = time.time()
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

    # ─── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="skill_curator.loop")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None

    def note_activity(self) -> None:
        self._last_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_activity) >= self._min_idle_minutes * 60

    async def _loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._interval_hours * 3600)
                if not self._running:
                    break
                if not self._is_idle():
                    continue
                try:
                    await self.run()
                except Exception as exc:  # noqa: BLE001
                    log.error("skill_curator.cycle_failed", error=str(exc))
        except asyncio.CancelledError:
            pass

    # ─── Core ───────────────────────────────────────────────────────

    async def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Plan + (unless dry-run) apply curator transitions."""
        entries = await self._registry.list_all()
        transitions = plan_transitions(
            entries,
            now=time.time(),
            stale_days=self._stale_days,
            archive_days=self._archive_days,
            min_success_rate=self._min_success_rate,
            min_runs=self._min_runs,
            managed_sources=self._managed_sources,
        )

        if not dry_run:
            for trans in transitions:
                try:
                    if trans.to_state == "archived":
                        await self._archive_skill(trans.name)
                    else:
                        await self._registry.set_state(trans.name, trans.to_state)
                        if trans.to_state == "active":
                            await self._registry.set_status(trans.name, "active")
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "skill_curator.apply_failed",
                        name=trans.name,
                        error=str(exc),
                    )

        report = {
            "dry_run": dry_run,
            "count": len(transitions),
            "transitions": [t.to_public() for t in transitions],
        }
        log.info(
            "skill_curator.run",
            dry_run=dry_run,
            transitions=len(transitions),
        )
        if not dry_run and transitions:
            try:
                await self._api.ws_broadcast(
                    {"type": "skill_curator_ran", **report}
                )
            except Exception:  # noqa: BLE001
                pass
        return report

    async def _archive_skill(self, name: str) -> None:
        """Move a skill's folder into ``.archive/`` and disable it."""
        entry = await self._registry.get(name)
        if entry is None:
            return
        folder = Path(entry.file_path)
        archive_root = self._skills_path / ".archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        dest = archive_root / f"{name}-{int(time.time())}"
        if folder.is_dir():
            try:
                await asyncio.to_thread(shutil.move, str(folder), str(dest))
                await self._registry.set_file_path(name, str(dest))
            except OSError as exc:
                log.warning(
                    "skill_curator.archive_move_failed", name=name, error=str(exc)
                )
        await self._registry.set_state(name, "archived")
        await self._registry.set_status(name, "disabled")
        log.info("skill_curator.archived", name=name)

    async def restore(self, name: str) -> bool:
        """Bring an archived skill back to active (folder + registry)."""
        entry = await self._registry.get(name)
        if entry is None:
            return False
        src = Path(entry.file_path)
        dest = self._skills_path / name
        if src.is_dir() and not dest.exists():
            try:
                await asyncio.to_thread(shutil.move, str(src), str(dest))
                await self._registry.set_file_path(name, str(dest))
            except OSError as exc:
                log.warning(
                    "skill_curator.restore_move_failed", name=name, error=str(exc)
                )
                return False
        await self._registry.set_state(name, "active")
        await self._registry.set_status(name, "active")
        log.info("skill_curator.restored", name=name)
        return True

    async def set_pinned(self, name: str, pinned: bool) -> bool:
        entry = await self._registry.get(name)
        if entry is None:
            return False
        await self._registry.set_pinned(name, pinned)
        return True

    async def status(self) -> dict[str, Any]:
        """Summary of the current lifecycle states for the UI."""
        entries = await self._registry.list_all()
        by_state: dict[str, int] = {}
        managed = 0
        for entry in entries:
            by_state[entry.state] = by_state.get(entry.state, 0) + 1
            if entry.source in self._managed_sources:
                managed += 1
        return {
            "total": len(entries),
            "managed": managed,
            "by_state": by_state,
            "thresholds": {
                "stale_days": self._stale_days,
                "archive_days": self._archive_days,
                "min_success_rate": self._min_success_rate,
                "min_runs": self._min_runs,
            },
        }
