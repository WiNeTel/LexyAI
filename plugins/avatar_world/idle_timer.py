"""
Avatar World — IdleTimer.

Tracks how long since the user was last seen and emits a coarse
``IdleStage`` enum: ``active`` → ``tired`` → ``sleeping``. The plugin reacts
to stage transitions by bumping emotion/activity.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum


class IdleStage(str, Enum):
    ACTIVE = "active"
    TIRED = "tired"
    SLEEPING = "sleeping"


class IdleTimer:
    """Computes the current IdleStage from a 'last seen' timestamp."""

    def __init__(
        self,
        tired_after_s: float = 600.0,
        sleep_after_s: float = 1800.0,
    ) -> None:
        # Defensive ordering: sleep threshold must be at least as large as
        # the tired threshold or the FSM gets stuck skipping 'tired'.
        self._tired_after_s = max(1.0, float(tired_after_s))
        self._sleep_after_s = max(self._tired_after_s, float(sleep_after_s))
        self._stage: IdleStage = IdleStage.ACTIVE

    @property
    def stage(self) -> IdleStage:
        return self._stage

    def mark_seen(self, now: float | None = None) -> IdleStage:
        """Reset the idle clock — the user just did something."""
        # ``now`` is accepted only so tests can pin a deterministic time.
        _ = now
        self._stage = IdleStage.ACTIVE
        return self._stage

    def tick(self, last_seen_at: float, now: float | None = None) -> IdleStage:
        """Recompute stage; return the (possibly unchanged) current stage."""
        ts_now = float(now) if now is not None else datetime.now().timestamp()
        elapsed = ts_now - float(last_seen_at)
        if elapsed >= self._sleep_after_s:
            self._stage = IdleStage.SLEEPING
        elif elapsed >= self._tired_after_s:
            self._stage = IdleStage.TIRED
        else:
            self._stage = IdleStage.ACTIVE
        return self._stage
