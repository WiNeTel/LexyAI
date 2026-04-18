"""Tests for ChannelRouter + SessionRouter wiring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lexy_core.channels.channel_base import ChannelBase, ChannelMessage
from lexy_core.channels.session_router import ChannelRouter, SessionRouter


class FakeChannel(ChannelBase):
    """Minimal in-memory channel for tests."""

    def __init__(self, name: str = "fake") -> None:
        super().__init__(name=name)
        self.sent: list[ChannelMessage] = []
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def send(self, message: ChannelMessage) -> None:
        self.sent.append(message)


def test_session_router_stable_id() -> None:
    router = SessionRouter()
    sid1 = router.get_session_id("discord", "123")
    sid2 = router.get_session_id("discord", "123")
    sid3 = router.get_session_id("discord", "456")
    sid4 = router.get_session_id("telegram", "123")
    assert sid1 == sid2
    assert sid1 != sid3
    assert sid1 != sid4
    assert sid1.startswith("sess-discord-")
    assert sid4.startswith("sess-telegram-")


def test_session_router_reset() -> None:
    router = SessionRouter()
    sid = router.get_session_id("telegram", "99")
    assert ("telegram", "99") in router.all_sessions()
    router.reset("telegram", "99")
    assert ("telegram", "99") not in router.all_sessions()


@pytest.mark.asyncio
async def test_channel_router_register_and_send() -> None:
    app = SimpleNamespace(
        agent=None,
        event_bus=SimpleNamespace(emit=AsyncMock()),
    )
    router = ChannelRouter(app)  # type: ignore[arg-type]
    channel = FakeChannel("discord")
    router.register(channel)

    assert "discord" in router.list_channels()
    assert router.get("discord") is channel

    msg = ChannelMessage(text="out", sender_id="42", channel="discord")
    await router.send("discord", msg)
    assert channel.sent == [msg]


@pytest.mark.asyncio
async def test_channel_router_send_unknown_raises() -> None:
    app = SimpleNamespace(agent=None, event_bus=SimpleNamespace(emit=AsyncMock()))
    router = ChannelRouter(app)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        await router.send("nope", ChannelMessage(text="x", sender_id="y", channel="nope"))


@pytest.mark.asyncio
async def test_channel_router_inbound_emits_event_and_calls_agent() -> None:
    agent = MagicMock()
    agent.process = AsyncMock(return_value={"text": "reply"})
    event_bus = SimpleNamespace(emit=AsyncMock())
    app = SimpleNamespace(agent=agent, event_bus=event_bus)
    router = ChannelRouter(app)  # type: ignore[arg-type]

    channel = FakeChannel("discord")
    router.register(channel)

    inbound = ChannelMessage(
        text="hello lexy",
        sender_id="user-42",
        channel="discord",
    )
    # Trigger the callback the same way a channel plugin would:
    await channel.emit_inbound(inbound)

    # Event was emitted
    assert event_bus.emit.await_count == 1
    name, payload = event_bus.emit.await_args.args
    assert name == "core.channel_message"
    assert payload["text"] == "hello lexy"
    assert payload["channel"] == "discord"

    # Agent.process was called with the resolved session id
    agent.process.assert_awaited_once()
    kwargs = agent.process.await_args.kwargs
    assert kwargs["text"] == "hello lexy"
    assert kwargs["session_id"].startswith("sess-discord-")

    # Reply was sent back through the channel
    assert len(channel.sent) == 1
    assert channel.sent[0].text == "reply"
    assert channel.sent[0].sender_id == "user-42"


@pytest.mark.asyncio
async def test_channel_router_inbound_without_agent_logs_and_returns() -> None:
    """If the agent isn't ready the router must not crash."""
    event_bus = SimpleNamespace(emit=AsyncMock())
    app = SimpleNamespace(agent=None, event_bus=event_bus)
    router = ChannelRouter(app)  # type: ignore[arg-type]
    channel = FakeChannel("discord")
    router.register(channel)

    await channel.emit_inbound(
        ChannelMessage(text="x", sender_id="y", channel="discord")
    )
    # Event still emitted, no reply sent
    assert event_bus.emit.await_count == 1
    assert channel.sent == []


def test_channel_router_unregister_removes() -> None:
    app = SimpleNamespace(agent=None, event_bus=SimpleNamespace(emit=AsyncMock()))
    router = ChannelRouter(app)  # type: ignore[arg-type]
    channel = FakeChannel("telegram")
    router.register(channel)
    assert "telegram" in router.list_channels()
    router.unregister("telegram")
    assert "telegram" not in router.list_channels()


def test_channel_router_unregister_unknown_is_noop() -> None:
    app = SimpleNamespace(agent=None, event_bus=SimpleNamespace(emit=AsyncMock()))
    router = ChannelRouter(app)  # type: ignore[arg-type]
    router.unregister("nope")  # should not raise
