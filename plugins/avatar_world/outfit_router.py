"""
Avatar World — OutfitRouter.

Picks an outfit id based on the current TimeOfDay + IdleStage, scoped to
the plugin's whitelist of available outfits. The router is pure: given
the same inputs it returns the same outfit. The plugin owns the side
effect of applying it to ``AvatarState``.

Defaults are tuned for the "junge Frau, freundlich-warm" character:
  - night + sleeping → pyjama
  - night            → casual (Lexy hangs out, not in bed yet)
  - evening          → casual
  - morning/midday   → casual (Phase 1: no business slot triggered yet)
  - explicit user request always wins (handled in the plugin, not here)
"""

from __future__ import annotations

from typing import Sequence

from plugins.avatar_world.idle_timer import IdleStage
from plugins.avatar_world.state import TimeOfDay


def suggest_outfit(
    bucket: TimeOfDay,
    idle_stage: IdleStage,
    available: Sequence[str],
    fallback: str = "casual",
) -> str:
    """Return an outfit id from ``available`` that fits the context.

    The caller is expected to compare the suggestion to the avatar's
    current outfit and only push an ``avatar.outfit`` event if it
    actually changes.
    """
    available_set = set(available)

    def _first(*candidates: str) -> str:
        for c in candidates:
            if c in available_set:
                return c
        # Fall back to whatever the plugin allows.
        return fallback if fallback in available_set else (
            next(iter(available)) if available else ""
        )

    if bucket == "night" and idle_stage == IdleStage.SLEEPING:
        return _first("pyjama", "casual")
    if bucket == "night":
        # Late evening but not yet asleep — still casual, not pyjama.
        return _first("casual", "pyjama")
    if bucket == "evening":
        return _first("casual")
    if bucket == "morning":
        return _first("casual")
    # midday / afternoon
    return _first("casual")
