"""
Avatar World — WebSocket publisher.

Thin wrapper around ``PluginAPI.ws_broadcast`` that keeps the WS payload
schema in one place. Every helper sends a single ``{"type": ..., "payload": ...}``
frame to all connected clients; the frontend's ``ws_listener.js``
dispatches by ``type``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lexy_core.plugin_system.plugin_api import PluginAPI


class WSPublisher:
    """Stateless helper bound to a single plugin's PluginAPI."""

    def __init__(self, api: "PluginAPI") -> None:
        self._api = api

    # ─── Generic ────────────────────────────────────────────────────

    async def send(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Send a single typed avatar.* frame to every WS client."""
        await self._api.ws_broadcast({"type": msg_type, "payload": payload})

    # ─── Typed helpers (one per WS topic) ───────────────────────────

    async def send_state(self, snapshot: dict[str, Any]) -> None:
        """Full state — emitted on connect + on big transitions."""
        await self.send("avatar.state", snapshot)

    async def send_emotion(
        self, name: str, intensity: float, ramp_ms: int = 600
    ) -> None:
        await self.send(
            "avatar.emotion",
            {"name": name, "intensity": round(intensity, 3), "ramp_ms": ramp_ms},
        )

    async def send_activity(
        self, activity_id: str, transition: str = "cut"
    ) -> None:
        await self.send(
            "avatar.activity",
            {"id": activity_id, "transition": transition},
        )

    async def send_speaking(self, state: str, stream_id: str = "") -> None:
        """``state`` is "start" or "end"."""
        await self.send(
            "avatar.speaking",
            {"state": state, "stream_id": stream_id},
        )

    async def send_outfit(self, outfit: str, reason: str = "") -> None:
        await self.send(
            "avatar.outfit",
            {"outfit": outfit, "reason": reason},
        )

    async def send_attention(self, look_at: str) -> None:
        await self.send(
            "avatar.attention",
            {"look_at": look_at},
        )

    async def send_background(self, background_id: str, fade_ms: int = 2000) -> None:
        await self.send(
            "avatar.background",
            {"id": background_id, "fade_ms": fade_ms},
        )

    async def send_view_mode(self, mode: str) -> None:
        await self.send(
            "avatar.view_mode",
            {"mode": mode},
        )
