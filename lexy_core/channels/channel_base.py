"""
Lexy AI - Channel Abstraction.

A channel is an external messaging surface (WhatsApp, Discord, Telegram, …).
Plugins implement ``ChannelBase`` and register with the ``ChannelRouter`` via
``PluginAPI.register_channel(...)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ChannelMessage(BaseModel):
    """Inbound or outbound channel message."""

    text: str
    sender_id: str
    channel: str  # whatsapp | discord | telegram | instagram | …
    metadata: dict[str, Any] = Field(default_factory=dict)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ChannelBase(ABC):
    """
    Base class for an external channel.

    Lifecycle
    ---------
    * ``connect()`` – open transport (WS/REST poller/bot).
    * ``send(message)`` – outbound delivery.
    * ``disconnect()`` – tear down transport.

    Plugins use the ``inbound`` callback (``set_inbound_callback``) to forward
    received messages to the LexyAgent via the ChannelRouter.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._inbound_cb: Any | None = None

    def set_inbound_callback(self, callback: Any) -> None:
        """Register the callback used for inbound messages."""
        self._inbound_cb = callback

    @abstractmethod
    async def connect(self) -> None:
        """Open the transport."""

    @abstractmethod
    async def send(self, message: ChannelMessage) -> None:
        """Deliver an outbound message to the recipient."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the transport."""

    async def emit_inbound(self, message: ChannelMessage) -> None:
        """Helper for subclasses to forward an inbound message upward."""
        if self._inbound_cb is not None:
            await self._inbound_cb(message)
