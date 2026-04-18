"""
Lexy AI - WhatsApp Channel Plugin.

Talks to a Node.js Baileys bridge over HTTP:

* ``GET  {bridge_url}{inbound_path}`` – long-poll for pending messages
  ``[{"jid": "...", "from_me": false, "text": "...", "id": "..."}]``
* ``POST {bridge_url}{outbound_path}`` – deliver an outbound message
  ``{"jid": "...", "text": "..."}``

Authentication is via a shared ``X-API-Key`` header (set through
``plugins.yaml``). The bridge implementation lives outside this repository.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from lexy_core.channels import ChannelBase, ChannelMessage
from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger

log = get_logger(module="whatsapp_channel")


class WhatsAppChannel(ChannelBase):
    """Baileys-backed WhatsApp channel."""

    def __init__(
        self,
        bridge_url: str,
        api_key: str,
        inbound_path: str,
        outbound_path: str,
        poll_interval: float,
        max_message_length: int,
        allowed_contacts: list[str],
    ) -> None:
        super().__init__(name="whatsapp")
        self._bridge_url = bridge_url.rstrip("/")
        self._api_key = api_key
        self._inbound_path = inbound_path
        self._outbound_path = outbound_path
        self._poll_interval = poll_interval
        self._max_len = max_message_length
        self._allowed = set(allowed_contacts or [])
        self._client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._running: bool = False
        # Rate-limited logging: warn once, then stay silent until the bridge
        # becomes reachable again. Stops flooding the log when the Node bridge
        # isn't running.
        self._bridge_down: bool = False
        self._consecutive_failures: int = 0

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"X-API-Key": self._api_key},
            timeout=60.0,
        )
        # Health probe — set initial state so the poll loop doesn't double-log.
        try:
            resp = await self._client.get(f"{self._bridge_url}/health")
            if resp.status_code >= 500:
                self._bridge_down = True
                log.warning("whatsapp.bridge_bad_health", status=resp.status_code)
        except httpx.HTTPError as exc:
            # Mark bridge as down silently; the poll loop will log once.
            self._bridge_down = True
            log.warning("whatsapp.bridge_unreachable", error=str(exc))

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("whatsapp.connected", bridge=self._bridge_url)

    async def disconnect(self) -> None:
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._poll_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        log.info("whatsapp.disconnected")

    async def send(self, message: ChannelMessage) -> None:
        if self._client is None:
            return
        text = message.text
        while text:
            chunk, text = text[: self._max_len], text[self._max_len :]
            try:
                resp = await self._client.post(
                    f"{self._bridge_url}{self._outbound_path}",
                    json={"jid": message.sender_id, "text": chunk},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.error("whatsapp.send_failed", error=str(exc))
                return

    async def _poll_loop(self) -> None:
        assert self._client is not None
        url = f"{self._bridge_url}{self._inbound_path}"
        while self._running:
            try:
                resp = await self._client.get(url)
                if resp.status_code == 200:
                    # Bridge is healthy again — log the recovery once.
                    if self._bridge_down:
                        log.info(
                            "whatsapp.bridge_recovered",
                            failures=self._consecutive_failures,
                        )
                    self._bridge_down = False
                    self._consecutive_failures = 0
                    payload = resp.json() or []
                    for item in payload:
                        if item.get("from_me"):
                            continue
                        sender = str(item.get("jid", ""))
                        if self._allowed and sender not in self._allowed:
                            log.warning("whatsapp.blocked_sender", sender=sender)
                            continue
                        await self.emit_inbound(
                            ChannelMessage(
                                text=str(item.get("text", "")),
                                sender_id=sender,
                                channel="whatsapp",
                                metadata={
                                    "message_id": item.get("id", ""),
                                    "timestamp": item.get("timestamp"),
                                },
                            )
                        )
                else:
                    self._register_failure(f"HTTP {resp.status_code}")
            except httpx.HTTPError as exc:
                self._register_failure(str(exc))
            except Exception as exc:  # noqa: BLE001
                log.error("whatsapp.poll_unexpected", error=str(exc))
            await asyncio.sleep(self._poll_interval)

    def _register_failure(self, reason: str) -> None:
        """Log the first failure, then stay silent until the bridge recovers."""
        self._consecutive_failures += 1
        if not self._bridge_down:
            self._bridge_down = True
            log.warning("whatsapp.bridge_unreachable", error=reason)


class WhatsAppChannelPlugin(BasePlugin):
    """Wire ``WhatsAppChannel`` into LexyApp."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._channel: WhatsAppChannel | None = None

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._channel = WhatsAppChannel(
            bridge_url=str(config.get("bridge_url", "http://127.0.0.1:3000")),
            api_key=str(config.get("api_key", "lexy-secret")),
            inbound_path=str(config.get("inbound_path", "/inbound")),
            outbound_path=str(config.get("outbound_path", "/send")),
            poll_interval=float(config.get("poll_interval", 2.0)),
            max_message_length=int(config.get("max_message_length", 4096)),
            allowed_contacts=list(config.get("allowed_contacts", []) or []),
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
