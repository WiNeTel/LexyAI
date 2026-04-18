"""
Lexy AI - Gemma4 4B Multimodal STT Plugin.

Uses the OpenAI-compatible ``/v1/audio/transcriptions`` endpoint exposed by a
dedicated llama.cpp server running the Gemma 4 4B multimodal checkpoint
(see ``architecture/services.md``).
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from lexy_core.plugin_system import BasePlugin
from lexy_core.utils.logging import get_logger
from lexy_core.voice.stt_base import STTProvider

log = get_logger(module="voice_gemma4")


class Gemma4STTProvider(STTProvider):
    """Remote STT via llama.cpp OpenAI-compatible endpoint."""

    name = "voice_gemma4"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str,
        language: str = "de",
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._language = language
        self._temperature = temperature
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._available: bool = False
        # Capability caching: None = unknown, True = confirmed working,
        # False = confirmed not supported by the llama.cpp build. Set on
        # the first call and never retried so we don't spam the log with
        # 404s / 500s on every mic click.
        self._audio_endpoint_supported: bool | None = None
        self._chat_audio_supported: bool | None = None

    async def initialize(self) -> bool:
        self._client = httpx.AsyncClient(timeout=self._timeout)
        base = self._endpoint.removesuffix("/v1")
        try:
            resp = await self._client.get(f"{base}/health")
            self._available = resp.status_code < 500
        except httpx.HTTPError as exc:
            log.warning("gemma4_stt.unreachable", endpoint=self._endpoint, error=str(exc))
            self._available = False

        if self._available:
            log.info(
                "gemma4_stt.ready",
                endpoint=self._endpoint,
                model=self._model,
                language=self._language,
            )
        return self._available

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 24000) -> str:
        if self._client is None or not self._available:
            return ""
        if not audio_bytes:
            return ""

        # If both capability probes have been negative in past calls, we
        # already know this build can't do STT — short-circuit without
        # firing another request and without logging.
        if (
            self._audio_endpoint_supported is False
            and self._chat_audio_supported is False
        ):
            return ""

        # Try /v1/audio/transcriptions first (unless we already know it's missing)
        if self._audio_endpoint_supported is not False:
            try:
                resp = await self._client.post(
                    f"{self._endpoint}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={
                        "file": ("audio.wav", audio_bytes, "audio/wav"),
                    },
                    data={
                        "model": self._model,
                        "language": self._language,
                        "temperature": str(self._temperature),
                        "response_format": "json",
                    },
                )
                if resp.status_code == 404:
                    # Endpoint doesn't exist on this build — cache and never retry
                    if self._audio_endpoint_supported is None:
                        log.warning(
                            "gemma4_stt.audio_endpoint_unavailable",
                            endpoint=self._endpoint,
                            hint="llama.cpp build has no /v1/audio/transcriptions; falling back to chat completions",
                        )
                    self._audio_endpoint_supported = False
                else:
                    resp.raise_for_status()
                    self._audio_endpoint_supported = True
                    data = resp.json()
                    text = str(data.get("text", "")).strip()
                    log.debug("gemma4_stt.transcribed", length=len(text))
                    return text
            except httpx.HTTPError as exc:
                # HTTP-level failure (not 404) — log once and move on
                if self._audio_endpoint_supported is None:
                    log.error("gemma4_stt.http_error", error=str(exc))
                self._audio_endpoint_supported = False

        # Fallback path: try chat completions with an audio content block
        return await self._fallback_chat(audio_bytes)

    async def _fallback_chat(self, audio_bytes: bytes) -> str:
        """
        Secondary path via /v1/chat/completions with a base64 audio input.
        Used when /v1/audio/transcriptions is not exposed by the llama.cpp build.
        """
        assert self._client is not None
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        messages = [
            {
                "role": "system",
                "content": f"Transcribe the audio to plain text in language={self._language}. Only output the transcript.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav"},
                    },
                ],
            },
        ]
        # Short-circuit if we already probed this path and it failed
        if self._chat_audio_supported is False:
            return ""

        try:
            resp = await self._client.post(
                f"{self._endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": self._temperature,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            # First failure → loud warning with actionable hint, cache it,
            # all subsequent failures → silent.
            if self._chat_audio_supported is None:
                log.warning(
                    "gemma4_stt.chat_audio_unavailable",
                    error=str(exc),
                    hint=(
                        "llama.cpp rejects input_audio content blocks. Your "
                        "mmproj is probably vision-only. Either install an "
                        "audio-capable mmproj or enable voice_canary for "
                        "local STT."
                    ),
                )
            self._chat_audio_supported = False
            return ""

        self._chat_audio_supported = True
        data = resp.json()
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            log.error("gemma4_stt.bad_response", error=str(exc))
            return ""

    @property
    def stt_capable(self) -> bool:
        """
        Best-effort hint to the VoiceManager: returns False once we know
        neither STT path works for this server.
        """
        if self._audio_endpoint_supported is False and self._chat_audio_supported is False:
            return False
        return True

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._available = False
        log.info("gemma4_stt.shutdown")


class Gemma4STTPlugin(BasePlugin):
    """Registers ``Gemma4STTProvider`` with the VoiceManager."""

    def __init__(self, api: Any, manifest: Any) -> None:
        super().__init__(api, manifest)
        self._provider: Gemma4STTProvider | None = None

    async def on_load(self) -> None:
        config = self.api.get_config()
        self._provider = Gemma4STTProvider(
            endpoint=str(config.get("endpoint", "http://127.0.0.1:5007/v1")),
            api_key=str(config.get("api_key", "sk-lexy-local")),
            model=str(config.get("model", "gemma-4-4b-it")),
            language=str(config.get("language", "de")),
            temperature=float(config.get("temperature", 0.0)),
            timeout=float(config.get("timeout", 60.0)),
        )
        ok = await self._provider.initialize()
        if not ok:
            log.warning("voice_gemma4.unavailable")
            self._provider = None

    async def on_enable(self) -> None:
        if self._provider is not None:
            self.api.register_voice_provider("stt", self._provider)

    async def on_disable(self) -> None:
        if self._provider is not None:
            await self._provider.shutdown()
            self._provider = None
