"""
Lexy AI - Telegram Channel Plugin.

Registers a ``TelegramChannel`` with the ChannelRouter. Uses
``python-telegram-bot`` in polling mode (no webhook required). The bot token
lives in an environment variable so it never hits disk.
"""

from __future__ import annotations

import os
from typing import Any

from lexy_core.channels import ChannelBase, ChannelMessage
from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="telegram_channel")

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        ContextTypes,
        MessageHandler,
        filters,
    )

    _HAS_TELEGRAM = True
except ImportError:  # pragma: no cover
    _HAS_TELEGRAM = False


class TelegramChannel(ChannelBase):
    """Telegram bot channel using polling."""

    def __init__(
        self,
        token: str,
        allowed_users: list[int] | None = None,
        max_message_length: int = 4096,
    ) -> None:
        super().__init__(name="telegram")
        self._token = token
        self._allowed_users = set(allowed_users or [])
        self._max_len = max_message_length
        self._app: Application | None = None

    async def connect(self) -> None:
        if not _HAS_TELEGRAM:
            log.error("telegram.missing_dependency")
            return
        self._app = Application.builder().token(self._token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        await self._app.initialize()
        await self._app.start()
        if self._app.updater is not None:
            await self._app.updater.start_polling()
        log.info("telegram.connected")

    async def disconnect(self) -> None:
        if self._app is None:
            return
        if self._app.updater is not None:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None
        log.info("telegram.disconnected")

    async def send(self, message: ChannelMessage) -> None:
        if self._app is None:
            return
        try:
            chat_id = int(message.sender_id)
        except ValueError:
            log.error("telegram.bad_chat_id", sender=message.sender_id)
            return
        text = message.text
        while text:
            chunk, text = text[: self._max_len], text[self._max_len :]
            await self._app.bot.send_message(chat_id=chat_id, text=chunk)

    async def _on_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message is None or update.effective_user is None:
            return
        user_id = update.effective_user.id
        if self._allowed_users and user_id not in self._allowed_users:
            log.warning("telegram.blocked_user", user_id=user_id)
            return

        text = update.message.text or ""
        message = ChannelMessage(
            text=text,
            sender_id=str(update.message.chat_id),
            channel="telegram",
            metadata={"user_id": user_id, "message_id": update.message.message_id},
        )
        await self.emit_inbound(message)


class TelegramChannelPlugin(BasePlugin):
    """Wire ``TelegramChannel`` into LexyApp."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._channel: TelegramChannel | None = None

    async def on_load(self) -> None:
        config = self.api.get_config()
        token_env = str(config.get("token_env", "LEXY_TELEGRAM_TOKEN"))
        token = os.environ.get(token_env, "")
        if not token:
            log.warning("telegram.no_token", env=token_env)
            return

        self._channel = TelegramChannel(
            token=token,
            allowed_users=list(config.get("allowed_users", []) or []),
            max_message_length=int(config.get("max_message_length", 4096)),
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
