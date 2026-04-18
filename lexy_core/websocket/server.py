"""
Lexy AI - WebSocket Server.

Holds active client sessions, dispatches typed messages to registered
handlers, and broadcasts events. Plugins register handlers via
``PluginAPI.register_ws_handler(msg_type, handler)``; built-in handlers
(chat / TTS / STT) are wired by ``LexyApp``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from lexy_core.utils.logging import get_logger

if TYPE_CHECKING:
    from lexy_core.app import LexyApp

log = get_logger(module="ws_server")

WSHandler = Callable[..., Any | Awaitable[Any]]


class WSClient:
    """Wrapper around a connected WebSocket."""

    def __init__(self, client_id: str, websocket: WebSocket) -> None:
        self.client_id = client_id
        self.websocket = websocket
        self.user_id: str = "default"
        self.session_id: str = f"sess-ws-{client_id[:8]}"
        # Binary audio buffer for the streaming STT path. Cleared after each
        # ``stt_end`` message.
        self.audio_buffer: bytearray = bytearray()

    async def send_json(self, data: dict[str, Any]) -> None:
        await self.websocket.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        await self.websocket.send_bytes(data)


class WSServer:
    """
    Stateful WebSocket multiplexer.

    Built-in message types (registered by LexyApp):
    * ``chat``              → LexyAgent.process_stream
    * ``stt``               → VoiceManager.transcribe
    * ``tts``               → VoiceManager.synthesize_streaming
    * ``get_signals``       → snapshot of LexySignals
    * ``get_plugins``       → loaded plugin list
    """

    def __init__(self, app: "LexyApp") -> None:
        self._app = app
        self._clients: dict[str, WSClient] = {}
        self._handlers: dict[str, WSHandler] = {}
        self._handler_sources: dict[str, list[str]] = defaultdict(list)
        self.frontend_modules: dict[str, dict[str, Any]] = {}

    # ─── Handler registration ───────────────────────────────────────

    def register_handler(
        self, msg_type: str, handler: WSHandler, source: str = "core"
    ) -> None:
        """Register a handler for a message type."""
        if msg_type in self._handlers:
            log.warning(
                "ws.handler_overwrite",
                msg_type=msg_type,
                old_source=self._handler_sources.get(msg_type),
                new_source=source,
            )
        self._handlers[msg_type] = handler
        self._handler_sources[msg_type].append(source)
        log.debug("ws.handler_registered", msg_type=msg_type, source=source)

    def unregister_handler(self, msg_type: str, source: str = "core") -> None:
        if msg_type in self._handlers:
            del self._handlers[msg_type]
            self._handler_sources.pop(msg_type, None)
            log.debug("ws.handler_unregistered", msg_type=msg_type, source=source)

    # ─── Frontend modules ───────────────────────────────────────────

    def register_frontend_module(self, module_id: str, config: dict[str, Any]) -> None:
        self.frontend_modules[module_id] = config

    def unregister_frontend_module(self, module_id: str) -> None:
        self.frontend_modules.pop(module_id, None)

    # ─── Client lifecycle ───────────────────────────────────────────

    async def accept(self, websocket: WebSocket) -> WSClient:
        await websocket.accept()
        client = WSClient(client_id=uuid.uuid4().hex, websocket=websocket)
        self._clients[client.client_id] = client
        log.info("ws.connected", client=client.client_id)
        await client.send_json(
            {
                "type": "welcome",
                "client_id": client.client_id,
                "session_id": client.session_id,
                "version": self._app.config.system.version,
            }
        )
        return client

    async def disconnect(self, client: WSClient) -> None:
        self._clients.pop(client.client_id, None)
        log.info("ws.disconnected", client=client.client_id)

    # ─── Dispatch loop ──────────────────────────────────────────────

    async def handle(self, websocket: WebSocket) -> None:
        """Top-level handler called from the FastAPI route.

        Accepts both text frames (JSON events) and binary frames. Binary
        frames are accumulated into ``client.audio_buffer`` and handed to
        the registered ``stt_chunk`` handler (if any) for intermediate
        processing; the full buffer is only delivered when a terminating
        ``{"type": "stt_end"}`` text frame arrives.
        """
        client = await self.accept(websocket)
        try:
            while True:
                try:
                    raw = await websocket.receive()
                except WebSocketDisconnect:
                    break
                if raw.get("type") == "websocket.disconnect":
                    break

                # Binary frame → accumulate into the client's audio buffer
                if raw.get("bytes") is not None:
                    client.audio_buffer.extend(raw["bytes"])
                    continue

                # Text frame → parse JSON and dispatch by message type
                text = raw.get("text")
                if text is None:
                    continue
                try:
                    import json as _json

                    message = _json.loads(text)
                except Exception:  # noqa: BLE001
                    await client.send_json(
                        {"type": "error", "error": "invalid JSON frame"}
                    )
                    continue
                if not isinstance(message, dict):
                    continue
                await self._dispatch(client, message)
        except Exception as exc:  # noqa: BLE001
            log.error("ws.loop_error", client=client.client_id, error=str(exc))
        finally:
            await self.disconnect(client)

    async def _dispatch(self, client: WSClient, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if not msg_type:
            await client.send_json({"type": "error", "error": "missing 'type' field"})
            return

        handler = self._handlers.get(msg_type)
        if handler is None:
            await client.send_json(
                {"type": "error", "error": f"unknown message type: {msg_type}"}
            )
            return

        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(client, message)
            else:
                handler(client, message)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "ws.handler_error",
                msg_type=msg_type,
                client=client.client_id,
                error=str(exc),
            )
            await client.send_json({"type": "error", "error": str(exc)})

    # ─── Send helpers ───────────────────────────────────────────────

    async def broadcast(self, data: dict[str, Any]) -> int:
        """Broadcast a message to every connected client. Returns count."""
        sent = 0
        for client in list(self._clients.values()):
            try:
                await client.send_json(data)
                sent += 1
            except Exception as exc:  # noqa: BLE001
                log.error("ws.broadcast_error", client=client.client_id, error=str(exc))
        return sent

    async def send_to(self, client_id: str, data: dict[str, Any]) -> bool:
        client = self._clients.get(client_id)
        if client is None:
            return False
        await client.send_json(data)
        return True
