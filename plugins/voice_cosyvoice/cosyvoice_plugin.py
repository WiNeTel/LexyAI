"""
Lexy AI - CosyVoice 3 TTS Plugin.

Registers the ``CosyVoiceTTS`` provider with the VoiceManager and exposes two
WebSocket handlers for runtime config tweaks.
"""

from __future__ import annotations

from typing import Any

from cosyvoice_tts import CosyVoiceTTS

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from lexy_core.websocket.server import WSClient

log = get_logger(module="cosyvoice_plugin")


class CosyVoicePlugin(BasePlugin):
    """CosyVoice 3 TTS (remote server)."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._tts: CosyVoiceTTS | None = None

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._tts = CosyVoiceTTS(
            server_url=str(config.get("endpoint", "http://172.20.0.245:5500")),
            voice=str(config.get("voice", "referenz_mio")),
            narrator_voice=str(config.get("narrator_voice", "referenz_mio")),
            speed=float(config.get("speed", 1.0)),
            timeout=float(config.get("timeout", 30.0)),
            retries=int(config.get("retries", 2)),
            streaming=bool(config.get("streaming", True)),
            default_instruct=str(config.get("default_instruct", "")),
            narrator_mode=str(config.get("narrator_mode", "full")),
            segment_pause_ms=int(config.get("segment_pause_ms", 80)),
        )
        ok = await self._tts.initialize()
        if not ok:
            log.warning("cosyvoice_plugin.server_unreachable")
            self._tts = None

    async def on_enable(self) -> None:
        if self._tts is not None:
            self.api.register_voice_provider("tts", self._tts)
            log.info("cosyvoice_plugin.registered")

        self.api.register_ws_handler("get_tts_config", self._handle_get_tts_config)
        self.api.register_ws_handler(
            "update_tts_config", self._handle_update_tts_config
        )

    async def _handle_get_tts_config(
        self, client: WSClient, message: dict[str, Any]
    ) -> None:
        config = self._tts.get_config() if self._tts is not None else {}
        await client.send_json({"type": "tts_config", "config": config})

    async def _handle_update_tts_config(
        self, client: WSClient, message: dict[str, Any]
    ) -> None:
        if self._tts is None:
            await client.send_json({"type": "error", "error": "TTS not available"})
            return
        self._tts.update_config(message.get("config", message))
        log.info("cosyvoice_plugin.config_updated")
        await client.send_json(
            {"type": "tts_config_updated", "config": self._tts.get_config()}
        )

    async def on_disable(self) -> None:
        if self._tts is not None:
            await self._tts.shutdown()
            self._tts = None
