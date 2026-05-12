"""
Avatar World — EventMapper.

Translates backend events (``core.user_message``, ``core.ai_response``,
``core.brain_routed`` …) into avatar reactions: emotion bumps, attention
shifts, idle resets. The plugin wires its event subscriptions through
this single entry point so the rules live in one readable file.

Phase 1 keeps the rule table small. Adding a new event here is the
intended way to grow the avatar's reactivity over time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lexy_core.utils.logging import get_logger
from plugins.avatar_world.idle_timer import IdleTimer
from plugins.avatar_world.state import AvatarState

if TYPE_CHECKING:
    from plugins.avatar_world.emotion_engine import EmotionEngine
    from plugins.avatar_world.ws_publisher import WSPublisher

log = get_logger(module="avatar_world.event_mapper")


class EventMapper:
    """Backend-event → avatar-reaction translator."""

    def __init__(
        self,
        state: AvatarState,
        emotion: "EmotionEngine",
        ws: "WSPublisher",
        idle: IdleTimer,
    ) -> None:
        self._state = state
        self._emotion = emotion
        self._ws = ws
        self._idle = idle

    # ─── Entry point — called from the plugin's event handler ───────

    async def apply(self, event_name: str, data: dict[str, Any]) -> None:
        """Apply an event to the avatar state. Silently no-ops on unknown
        events so we don't have to whitelist every core.* topic at the
        subscription level — the bus delivers everything in ``core.*``
        and this method picks what to react to."""
        try:
            if event_name == "core.user_message":
                await self._on_user_message(data)
            elif event_name == "core.ai_response":
                await self._on_ai_response(data)
            elif event_name == "core.brain_routed":
                await self._on_brain_routed(data)
            elif event_name == "core.system_shutdown":
                await self._on_shutdown(data)
        except Exception as exc:  # noqa: BLE001 — never break the bus
            log.error(
                "avatar_world.event_mapper_error",
                event=event_name,
                error=str(exc),
            )

    # ─── Per-event handlers ─────────────────────────────────────────

    async def _on_user_message(self, data: dict[str, Any]) -> None:
        """User just sent something — Lexy notices and turns toward camera."""
        self._idle.mark_seen()
        self._state.last_user_seen_at = self._now()
        # Small surprised lift — "oh, hi!" — fades within ~30s by decay.
        emo = self._emotion.bump("surprised", 0.35)
        await self._ws.send_attention("camera")
        if self._state.gaze_target != "camera":
            self._state.gaze_target = "camera"
        await self._ws.send_emotion(emo.name, emo.intensity)

    async def _on_ai_response(self, data: dict[str, Any]) -> None:
        """Lexy just finished her answer — a touch of happy."""
        # Don't bump too hard — the answer is the deliverable, not a
        # celebration. The frontend lip-sync handles the actual speech.
        emo = self._emotion.bump("happy", 0.4)
        await self._ws.send_emotion(emo.name, emo.intensity)

    async def _on_brain_routed(self, data: dict[str, Any]) -> None:
        """Brain chose a route — Lexy is now actively thinking."""
        emo = self._emotion.bump("thinking", 0.6)
        # Looking at the (imaginary) monitor while she works.
        if self._state.gaze_target != "screen":
            self._state.gaze_target = "screen"
        await self._ws.send_emotion(emo.name, emo.intensity)

    async def _on_shutdown(self, data: dict[str, Any]) -> None:
        """System is going down — Lexy waves goodbye."""
        emo = self._emotion.force("tired", 0.4)
        await self._ws.send_emotion(emo.name, emo.intensity)

    # ─── Utilities ──────────────────────────────────────────────────

    @staticmethod
    def _now() -> float:
        from datetime import datetime

        return datetime.now().timestamp()
