"""
Avatar World — Pydantic state.

Single source of truth for what the avatar is currently doing/feeling. The
plugin mutates this in place; ``snapshot()`` produces a JSON-serialisable
dict that ``WSPublisher`` ships to the frontend.

Naming convention for fields matches the WS payload schema (so a snapshot
can be sent verbatim).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EmotionName = Literal["neutral", "happy", "thinking", "surprised", "tired"]
TimeOfDay = Literal["morning", "midday", "afternoon", "evening", "night"]
GazeTarget = Literal["camera", "screen", "window", "ambient", "none"]
ViewMode = Literal["conversation", "ambient", "pip"]
Activity = Literal[
    "sit_desk",
    "sit_couch",
    "stand_window",
    "walk",
    "sleep_couch",
    "bathroom",
]


class EmotionState(BaseModel):
    """Currently dominant emotion plus its intensity."""

    name: EmotionName = "neutral"
    intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Wall-clock time when intensity was last bumped — used by the decay
    # tick to know how much to fade.
    updated_at: float = Field(default_factory=lambda: datetime.now().timestamp())


class AvatarState(BaseModel):
    """Full snapshot of avatar state shared with the frontend."""

    emotion: EmotionState = Field(default_factory=EmotionState)
    activity: Activity = "sit_desk"
    outfit: str = "casual"
    outfit_variant: str = "default"
    time_of_day: TimeOfDay = "midday"
    background_id: str = "city_day"
    gaze_target: GazeTarget = "ambient"
    view_mode_hint: ViewMode = "ambient"

    # Last time the avatar saw "the user is here" (user message, STT,
    # explicit attention event). Drives the idle progression.
    last_user_seen_at: float = Field(
        default_factory=lambda: datetime.now().timestamp()
    )

    def snapshot(self) -> dict:
        """JSON-friendly dict suitable for an ``avatar.state`` WS payload."""
        return {
            "emotion": {
                "name": self.emotion.name,
                "intensity": round(self.emotion.intensity, 3),
            },
            "activity": self.activity,
            "outfit": self.outfit,
            "outfit_variant": self.outfit_variant,
            "time_of_day": self.time_of_day,
            "background_id": self.background_id,
            "gaze_target": self.gaze_target,
            "view_mode_hint": self.view_mode_hint,
        }
