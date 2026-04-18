"""
Lexy AI - Channel + Session Routing.

* ``ChannelRouter`` keeps a registry of active channels and dispatches inbound
  messages to LexyAgent. Plugins register channels via PluginAPI.

* ``SessionRouter`` maps a (channel, sender_id) pair to a stable session_id so
  conversations across re-connects share the same memory context.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

from lexy_core.channels.channel_base import ChannelBase, ChannelMessage
from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.app import LexyApp

log = get_logger(module="channel_router")


class SessionRouter:
    """
    Map ``(channel, sender_id)`` → stable ``session_id``.

    Sessions live in process memory; restart wipes them. Plugins that need
    persistence can save/restore the mapping themselves.
    """

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], str] = {}

    def get_session_id(self, channel: str, sender_id: str) -> str:
        key = (channel, sender_id)
        if key not in self._sessions:
            seed = hashlib.sha1(f"{channel}:{sender_id}".encode()).hexdigest()[:12]
            self._sessions[key] = f"sess-{channel}-{seed}"
        return self._sessions[key]

    def reset(self, channel: str, sender_id: str) -> None:
        self._sessions.pop((channel, sender_id), None)

    def all_sessions(self) -> dict[tuple[str, str], str]:
        return dict(self._sessions)


class ChannelRouter:
    """
    Active channel registry. Inbound messages are forwarded to the LexyAgent.
    """

    def __init__(self, app: "LexyApp") -> None:
        self._app = app
        self._channels: dict[str, ChannelBase] = {}
        self._sessions = SessionRouter()

    @property
    def sessions(self) -> SessionRouter:
        return self._sessions

    def get(self, name: str) -> ChannelBase | None:
        return self._channels.get(name)

    def list_channels(self) -> list[str]:
        return list(self._channels)

    def register(self, channel: ChannelBase) -> None:
        """Register a channel and wire its inbound callback."""
        self._channels[channel.name] = channel
        channel.set_inbound_callback(self._on_inbound)
        log.info("channel.registered", channel=channel.name)

    def unregister(self, name: str) -> None:
        channel = self._channels.pop(name, None)
        if channel is not None:
            log.info("channel.unregistered", channel=name)

    async def send(self, channel: str, message: ChannelMessage) -> None:
        """Send an outbound message via the named channel."""
        target = self._channels.get(channel)
        if target is None:
            raise KeyError(f"Channel '{channel}' not registered")
        await target.send(message)

    async def _on_inbound(self, message: ChannelMessage) -> None:
        """
        Handle an inbound message. Resolves a session id and forwards the
        text to the LexyAgent. The agent's response is sent back through the
        same channel.
        """
        session_id = self._sessions.get_session_id(message.channel, message.sender_id)
        log.info(
            "channel.inbound",
            channel=message.channel,
            sender=message.sender_id,
            session=session_id,
            length=len(message.text),
        )

        await self._app.event_bus.emit(
            "core.channel_message",
            {
                "text": message.text,
                "sender": message.sender_id,
                "channel": message.channel,
                "session_id": session_id,
            },
        )

        if self._app.agent is None:
            log.warning("channel.no_agent")
            return

        result = await self._app.agent.process(
            text=message.text,
            session_id=session_id,
            user_id=message.sender_id,
        )

        reply = ChannelMessage(
            text=result.get("text", ""),
            sender_id=message.sender_id,
            channel=message.channel,
            metadata={"session_id": session_id, "request_id": uuid.uuid4().hex[:8]},
        )
        try:
            await self.send(message.channel, reply)
        except Exception as exc:  # noqa: BLE001
            log.error("channel.send_failed", channel=message.channel, error=str(exc))
