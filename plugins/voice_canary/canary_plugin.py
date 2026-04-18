"""
Lexy AI - NVIDIA Canary-1B-Flash STT Fallback.

Loads the Canary model via ``nemo_toolkit`` in a worker thread the first time
it is needed. If NeMo is not installed the plugin stays inactive and lets the
Gemma 4 plugin be the sole STT provider. This keeps the environment footprint
small when the user doesn't need local STT.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from lexy_core.voice.stt_base import STTProvider

log = get_logger(module="voice_canary")


class CanarySTTProvider(STTProvider):
    """Local STT using NVIDIA Canary-1B-Flash via NeMo."""

    name = "voice_canary"

    def __init__(self, model: str, device: str, language: str) -> None:
        self._model_name = model
        self._device = device
        self._language = language
        self._model: Any = None
        self._available: bool = False

    async def initialize(self) -> bool:
        try:
            self._model = await asyncio.to_thread(self._load)
        except ImportError as exc:
            log.warning("canary.nemo_missing", error=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            log.error("canary.load_failed", error=str(exc))
            return False
        self._available = True
        log.info("canary.ready", model=self._model_name, device=self._device)
        return True

    def _load(self) -> Any:
        from nemo.collections.asr.models import EncDecMultiTaskModel  # type: ignore[import]

        model = EncDecMultiTaskModel.from_pretrained(self._model_name)
        model = model.to(self._device)
        model.eval()
        return model

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 24000) -> str:
        if not self._available or self._model is None:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        import tempfile
        import os

        # NeMo expects a file path; write a small temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            # If the caller gave us raw PCM, wrap it in a minimal WAV header
            if audio_bytes[:4] != b"RIFF":
                with wave.open(handle.name, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(audio_bytes)
            else:
                handle.write(audio_bytes)
            path = handle.name

        try:
            result = self._model.transcribe(
                audio=[path],
                source_lang=self._language,
                target_lang=self._language,
                task="asr",
                pnc="yes",
            )
            if isinstance(result, list) and result:
                first = result[0]
                if isinstance(first, str):
                    return first.strip()
                if hasattr(first, "text"):
                    return str(first.text).strip()
            return ""
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def shutdown(self) -> None:
        self._model = None
        self._available = False
        log.info("canary.shutdown")


class CanarySTTPlugin(BasePlugin):
    """Registers Canary STT as a fallback with the VoiceManager."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._provider: CanarySTTProvider | None = None

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._provider = CanarySTTProvider(
            model=str(config.get("model", "nvidia/canary-1b-flash")),
            device=str(config.get("device", "cuda:0")),
            language=str(config.get("language", "de")),
        )
        if not await self._provider.initialize():
            self._provider = None

    async def on_enable(self) -> None:
        if self._provider is not None:
            self.api.register_voice_provider("stt", self._provider)

    async def on_disable(self) -> None:
        if self._provider is not None:
            await self._provider.shutdown()
            self._provider = None
