"""
Lexy AI - Discord Channel Plugin.

Uses ``discord.py`` to run a bot that listens for messages starting with the
configured prefix and forwards them to the LexyAgent via the ChannelRouter.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from lexy_core.channels import ChannelBase, ChannelMessage
from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="discord_channel")

try:
    import discord

    _HAS_DISCORD = True
except ImportError:  # pragma: no cover
    _HAS_DISCORD = False


class DiscordChannel(ChannelBase):
    """Discord bot channel."""

    def __init__(
        self,
        token: str,
        command_prefix: str,
        allowed_guilds: list[int] | None = None,
        allowed_channels: list[int] | None = None,
        max_message_length: int = 2000,
    ) -> None:
        super().__init__(name="discord")
        self._token = token
        self._prefix = command_prefix
        self._allowed_guilds = set(allowed_guilds or [])
        self._allowed_channels = set(allowed_channels or [])
        self._max_len = max_message_length
        self._client: discord.Client | None = None
        self._runner_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        if not _HAS_DISCORD:
            log.error("discord.missing_dependency")
            return

        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)

        @client.event
        async def on_message(message: discord.Message) -> None:  # type: ignore[name-defined]
            if message.author.bot:
                return
            if self._allowed_guilds and (
                message.guild is None or message.guild.id not in self._allowed_guilds
            ):
                return
            if self._allowed_channels and message.channel.id not in self._allowed_channels:
                return
            if not message.content.startswith(self._prefix):
                return

            text = message.content[len(self._prefix) :].strip()
            if not text:
                return

            await self.emit_inbound(
                ChannelMessage(
                    text=text,
                    sender_id=str(message.channel.id),
                    channel="discord",
                    metadata={
                        "user_id": message.author.id,
                        "username": str(message.author),
                        "guild_id": message.guild.id if message.guild else None,
                        "message_id": message.id,
                    },
                )
            )

        self._client = client
        self._runner_task = asyncio.create_task(client.start(self._token))
        log.info("discord.connected", prefix=self._prefix)

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._runner_task is not None and not self._runner_task.done():
            self._runner_task.cancel()
            try:
                await self._runner_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._runner_task = None
        log.info("discord.disconnected")

    async def send(self, message: ChannelMessage) -> None:
        if self._client is None:
            return
        try:
            channel_id = int(message.sender_id)
        except ValueError:
            log.error("discord.bad_channel_id", sender=message.sender_id)
            return
        channel = self._client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(channel_id)
            except Exception as exc:  # noqa: BLE001
                log.error("discord.fetch_channel_failed", error=str(exc))
                return

        text = message.text
        while text:
            chunk, text = text[: self._max_len], text[self._max_len :]
            await channel.send(chunk)  # type: ignore[union-attr]


class DiscordChannelPlugin(BasePlugin):
    """Wire ``DiscordChannel`` into LexyApp."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._channel: DiscordChannel | None = None

    async def on_load(self) -> None:
        config = self.api.get_config()
        token_env = str(config.get("token_env", "LEXY_DISCORD_TOKEN"))
        token = os.environ.get(token_env, "")
        if not token:
            log.warning("discord.no_token", env=token_env)
            return

        self._channel = DiscordChannel(
            token=token,
            command_prefix=str(config.get("command_prefix", "!lexy")),
            allowed_guilds=list(config.get("allowed_guilds", []) or []),
            allowed_channels=list(config.get("allowed_channels", []) or []),
            max_message_length=int(config.get("max_message_length", 2000)),
        )

    async def on_enable(self) -> None:
        if self._channel is None:
            return
        self.api.register_channel(self._channel)
        await self._channel.connect()

    async def on_disable(self) -> None:
        if self._channel is None:
            return
        await self._channel.disconnect()
        self._channel = None
