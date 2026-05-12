"""
Avatar World — EmotionEngine.

5 states (neutral / happy / thinking / surprised / tired) with intensity
0.0-1.0. Triggers ``bump(name, delta)`` increase intensity; the periodic
``decay(dt)`` call eases the intensity back toward 0 — when it drops below
a small floor we snap back to neutral.

Phase 1 keeps things simple: one dominant emotion at a time, no blending.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from plugins.avatar_world.state import AvatarState, EmotionName, EmotionState


class EmotionEngine:
    """Owns the avatar's emotion sub-state."""

    # Below this intensity floor we revert to 'neutral' so the avatar
    # doesn't permanently sit at "happy 0.01".
    _FLOOR = 0.05

    def __init__(
        self,
        state: AvatarState,
        decay_seconds: float = 30.0,
        on_change: Callable[[EmotionState], None] | None = None,
    ) -> None:
        self._state = state
        self._decay_seconds = max(0.5, float(decay_seconds))
        self._on_change = on_change

    # ─── Triggers ───────────────────────────────────────────────────

    def bump(self, name: EmotionName, delta: float) -> EmotionState:
        """Push intensity toward ``delta`` for the given emotion.

        If the new emotion differs from the current one we *replace*
        (Phase 1 has no blending). If it's the same emotion we add the
        delta, capped at 1.0. ``on_change`` fires when either the name
        or the intensity actually moved.
        """
        delta = max(0.0, min(1.0, float(delta)))
        current = self._state.emotion
        new_name = name
        if current.name == name:
            new_intensity = min(1.0, current.intensity + delta)
        else:
            new_intensity = delta

        if new_intensity < self._FLOOR:
            new_name = "neutral"
            new_intensity = 0.0

        changed = (new_name != current.name) or (
            abs(new_intensity - current.intensity) > 0.01
        )
        self._state.emotion = EmotionState(
            name=new_name,
            intensity=new_intensity,
            updated_at=datetime.now().timestamp(),
        )
        if changed and self._on_change is not None:
            self._on_change(self._state.emotion)
        return self._state.emotion

    def force(self, name: EmotionName, intensity: float) -> EmotionState:
        """Hard-set emotion (used by manual overrides from the UI)."""
        intensity = max(0.0, min(1.0, float(intensity)))
        if intensity < self._FLOOR:
            name = "neutral"
            intensity = 0.0
        self._state.emotion = EmotionState(
            name=name,
            intensity=intensity,
            updated_at=datetime.now().timestamp(),
        )
        if self._on_change is not None:
            self._on_change(self._state.emotion)
        return self._state.emotion

    # ─── Tick ───────────────────────────────────────────────────────

    def decay(self, dt: float) -> EmotionState | None:
        """Fade intensity over ``dt`` seconds. Returns new state on change.

        The decay is linear in the configured ``decay_seconds`` window:
        a fresh ``bump(..., 1.0)`` reaches 0 after ``decay_seconds``.
        Neutral emotions don't decay (they have nothing to fade to).
        """
        current = self._state.emotion
        if current.name == "neutral" or current.intensity <= 0.0:
            return None

        step = float(dt) / self._decay_seconds
        new_intensity = max(0.0, current.intensity - step)
        new_name: EmotionName = current.name
        if new_intensity < self._FLOOR:
            new_name = "neutral"
            new_intensity = 0.0

        if (
            abs(new_intensity - current.intensity) < 0.005
            and new_name == current.name
        ):
            return None  # below the change threshold, skip the WS push

        self._state.emotion = EmotionState(
            name=new_name,
            intensity=new_intensity,
            updated_at=current.updated_at,
        )
        if self._on_change is not None:
            self._on_change(self._state.emotion)
        return self._state.emotion
