"""
Lexy AI - CosyVoice 3 TTS Plugin.

Registers the ``CosyVoiceTTS`` provider with the VoiceManager and exposes two
WebSocket handlers for runtime config tweaks.

The ``CosyVoiceTTS`` implementation lives in this same package
(``plugins/voice_cosyvoice/cosyvoice_tts.py``) — we use a *relative*
import so the module resolves regardless of how the plugin loader sets
up ``sys.path``. The earlier top-level form (``from cosyvoice_tts``)
broke on machines where ``plugins/`` was on the path but the plugin's
own directory wasn't.
"""

from __future__ import annotations

import time
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from lexy_core.websocket.server import WSClient

log = get_logger(module="cosyvoice_plugin")


class CosyVoicePlugin(BasePlugin):
    """CosyVoice 3 TTS (remote server).

    The plugin is **always loadable** — even when the local TTS module
    can't be imported or the remote server is unreachable. ``on_load``
    catches every failure and stores it in ``self._last_error`` so
    ``get_status()`` can surface it to the UI; the plugin then runs in
    a degraded state with no provider registered. This avoids the 500
    Mike was seeing when clicking "Enable" — the loader no longer
    blows up on a missing import.
    """

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._tts: Any = None
        self._last_error: str = ""
        self._server_url: str = ""
        self._initialized_at: float = 0.0

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._server_url = str(
            config.get("endpoint", "http://172.20.0.245:5500")
        )

        # Lazy + relative import so the plugin remains loadable even if
        # the TTS module is missing or itself fails to import (e.g. one
        # of its own deps isn't installed on this machine). The previous
        # top-level ``from cosyvoice_tts import CosyVoiceTTS`` form
        # crashed the whole load step on those machines.
        try:
            from .cosyvoice_tts import CosyVoiceTTS  # type: ignore[import-not-found]
        except ImportError as exc:
            self._last_error = f"cosyvoice_tts module not importable: {exc}"
            log.warning(
                "cosyvoice_plugin.module_missing error=%s "
                "(plugin will run in degraded mode)",
                exc,
            )
            return
        except Exception as exc:  # noqa: BLE001 — anything else is a defect we want surfaced
            self._last_error = f"cosyvoice_tts import failed: {exc}"
            log.error("cosyvoice_plugin.import_failed error=%s", exc)
            return

        try:
            self._tts = CosyVoiceTTS(
                server_url=self._server_url,
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
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"CosyVoiceTTS construction failed: {exc}"
            log.error("cosyvoice_plugin.ctor_failed error=%s", exc)
            self._tts = None
            return

        try:
            ok = await self._tts.initialize()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"initialize() raised: {exc}"
            log.warning("cosyvoice_plugin.init_raised error=%s", exc)
            self._tts = None
            return

        if not ok:
            self._last_error = (
                f"server unreachable at {self._server_url}"
            )
            log.warning(
                "cosyvoice_plugin.server_unreachable url=%s",
                self._server_url,
            )
            self._tts = None
            return

        self._initialized_at = time.time()
        self._last_error = ""

    async def on_enable(self) -> None:
        if self._tts is not None:
            self.api.register_voice_provider("tts", self._tts)
            log.info(
                "cosyvoice_plugin.registered url=%s", self._server_url,
            )
        else:
            log.info(
                "cosyvoice_plugin.enabled_degraded last_error=%s",
                self._last_error or "(unknown)",
            )

        self.api.register_ws_handler("get_tts_config", self._handle_get_tts_config)
        self.api.register_ws_handler(
            "update_tts_config", self._handle_update_tts_config
        )

    def get_status(self) -> dict[str, Any]:
        """Status snapshot used by ``/api/v1/plugins/voice_cosyvoice/status``.

        Resolves the inconsistency Mike saw between the dashboard's
        green dot (= remote server reachable) and the plugin tab's
        "offline" badge (= module missing locally). Both are now
        reported separately under unambiguous names.
        """
        return {
            "tts_provider_active": self._tts is not None,
            "server_url": self._server_url,
            "module_importable": self._last_error == "" or
                "module not importable" not in self._last_error,
            "initialized_at": self._initialized_at,
            "last_error": self._last_error,
        }

    async def _handle_get_tts_config(
        self, client: WSClient, message: dict[str, Any]
    ) -> None:
        config = self._tts.get_config() if self._tts is not None else {}
        await client.send_json(
            {
                "type": "tts_config",
                "config": config,
                "available": self._tts is not None,
                "last_error": self._last_error,
            }
        )

    async def _handle_update_tts_config(
        self, client: WSClient, message: dict[str, Any]
    ) -> None:
        if self._tts is None:
            await client.send_json(
                {
                    "type": "error",
                    "error": (
                        f"TTS not available: {self._last_error}"
                        if self._last_error
                        else "TTS not available"
                    ),
                }
            )
            return
        self._tts.update_config(message.get("config", message))
        log.info("cosyvoice_plugin.config_updated")
        await client.send_json(
            {"type": "tts_config_updated", "config": self._tts.get_config()}
        )

    async def on_disable(self) -> None:
        if self._tts is not None:
            try:
                await self._tts.shutdown()
            except Exception as exc:  # noqa: BLE001
                log.warning("cosyvoice_plugin.shutdown_failed error=%s", exc)
            self._tts = None
