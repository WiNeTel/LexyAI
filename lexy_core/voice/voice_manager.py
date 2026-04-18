"""
Lexy AI - VoiceManager.

Provider-agnostic STT/TTS hub. Plugins register concrete providers via
``PluginAPI.register_voice_provider(kind, provider)``. The manager keeps a
primary provider per kind plus a fallback chain.

The configured primary names live in ``config.voice.stt.primary`` /
``config.voice.tts.primary``; if registration order differs, the configured
name wins. The fallback (``config.voice.stt.fallback``) is used if the primary
fails or is missing.
"""

from __future__ import annotations

from typing import AsyncGenerator

from lexy_core.config import VoiceConfig
from lexy_core.utils.logging import get_logger
from lexy_core.voice.stt_base import STTProvider
from lexy_core.voice.tts_base import TTSProvider

log = get_logger(module="voice_manager")


class VoiceManager:
    """
    Voice pipeline coordinator.

    Plugins register STT/TTS providers; the manager picks the configured
    primary or falls back to the fallback name. Multiple providers per kind
    may coexist; only one is active at a time.
    """

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._stt_providers: dict[str, STTProvider] = {}
        self._tts_providers: dict[str, TTSProvider] = {}

    # ─── Properties ─────────────────────────────────────────────────

    @property
    def has_stt(self) -> bool:
        return bool(self._stt_providers)

    @property
    def has_tts(self) -> bool:
        return bool(self._tts_providers)

    @property
    def supports_streaming(self) -> bool:
        active = self._select_tts()
        return active is not None and active.supports_streaming

    # ─── Registration ───────────────────────────────────────────────

    def register_provider(self, kind: str, name: str, provider: object) -> None:
        """Register a provider under a given kind ('stt' or 'tts')."""
        if kind == "stt":
            if not isinstance(provider, STTProvider):
                raise TypeError(
                    f"STT provider must subclass STTProvider, got {type(provider).__name__}"
                )
            self._stt_providers[name] = provider
            log.info("voice.stt_registered", name=name, type=type(provider).__name__)
        elif kind == "tts":
            if not isinstance(provider, TTSProvider):
                raise TypeError(
                    f"TTS provider must subclass TTSProvider, got {type(provider).__name__}"
                )
            self._tts_providers[name] = provider
            log.info("voice.tts_registered", name=name, type=type(provider).__name__)
        else:
            raise ValueError(f"Unknown voice provider kind: {kind!r}")

    def unregister_provider(self, kind: str, name: str) -> None:
        """Remove a provider previously registered with the given name."""
        if kind == "stt":
            if name in self._stt_providers:
                del self._stt_providers[name]
                log.info("voice.stt_unregistered", name=name)
        elif kind == "tts":
            if name in self._tts_providers:
                del self._tts_providers[name]
                log.info("voice.tts_unregistered", name=name)

    # ─── Selection ──────────────────────────────────────────────────

    def _select_stt(self) -> STTProvider | None:
        """
        Pick the active STT provider. Skips providers whose
        ``stt_capable`` flipped to False at runtime (e.g. gemma4 when
        the llama.cpp build doesn't expose an audio path) so we don't
        waste a round trip before falling back.
        """
        primary = self._config.stt.primary
        fallback = self._config.stt.fallback
        prov = self._stt_providers.get(primary)
        if prov is not None and prov.stt_capable:
            return prov
        prov = self._stt_providers.get(fallback)
        if prov is not None and prov.stt_capable:
            return prov
        # Final resort: any registered provider that is still capable
        for candidate in self._stt_providers.values():
            if candidate.stt_capable:
                return candidate
        # Nothing capable — return whatever's there so the caller gets a
        # consistent empty transcript instead of a "no provider" error.
        if primary in self._stt_providers:
            return self._stt_providers[primary]
        if fallback in self._stt_providers:
            return self._stt_providers[fallback]
        if self._stt_providers:
            return next(iter(self._stt_providers.values()))
        return None

    def _select_tts(self) -> TTSProvider | None:
        primary = self._config.tts.primary
        if primary in self._tts_providers:
            return self._tts_providers[primary]
        if self._tts_providers:
            return next(iter(self._tts_providers.values()))
        return None

    # ─── Operations ─────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int | None = None,
    ) -> str:
        """
        Transcribe audio. Tries the selected provider first; on an empty
        result OR an exception it automatically tries the configured
        fallback so the mic path keeps working when gemma4 can't
        handle audio (vision-only mmproj).
        """
        sr = sample_rate if sample_rate is not None else self._config.sample_rate
        provider = self._select_stt()
        if provider is None:
            log.warning("voice.no_stt_provider")
            return ""

        # Try primary
        text = ""
        failed_exc: Exception | None = None
        try:
            text = await provider.transcribe(audio_bytes, sample_rate=sr)
        except Exception as exc:  # noqa: BLE001 — try fallback below
            log.error(
                "voice.stt_failed",
                provider=getattr(provider, "name", type(provider).__name__),
                error=str(exc),
            )
            failed_exc = exc

        if text:
            return text

        # Empty/failed → fall back. Pick any *other* registered provider
        # (try the configured fallback first, then anything else).
        fallback_name = self._config.stt.fallback
        ordered: list[STTProvider] = []
        fb = self._stt_providers.get(fallback_name)
        if fb is not None and fb is not provider:
            ordered.append(fb)
        for name, candidate in self._stt_providers.items():
            if candidate is provider or candidate in ordered:
                continue
            ordered.append(candidate)

        for candidate in ordered:
            if not candidate.stt_capable:
                continue
            log.info(
                "voice.stt_fallback",
                fallback=getattr(candidate, "name", type(candidate).__name__),
                reason="exception" if failed_exc is not None else "empty",
            )
            try:
                fb_text = await candidate.transcribe(audio_bytes, sample_rate=sr)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "voice.stt_fallback_failed",
                    fallback=getattr(candidate, "name", type(candidate).__name__),
                    error=str(exc),
                )
                continue
            if fb_text:
                return fb_text
        return ""

    async def synthesize(
        self, text: str, voice: str | None = None
    ) -> bytes:
        """Synthesize text to a single WAV blob.

        ``voice`` is an optional provider-specific voice override —
        e.g. a CosyVoice speaker id per RP character. Providers that
        don't support multiple voices ignore it and use their default.
        """
        provider = self._select_tts()
        if provider is None:
            log.warning("voice.no_tts_provider")
            return b""
        return await provider.synthesize(text, voice=voice)

    async def synthesize_streaming(
        self, text: str, voice: str | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Streaming synthesis. Falls back to single-shot if unsupported."""
        provider = self._select_tts()
        if provider is None:
            log.warning("voice.no_tts_provider")
            return
        if provider.supports_streaming:
            async for chunk in provider.synthesize_streaming(text, voice=voice):
                yield chunk
        else:
            data = await provider.synthesize(text, voice=voice)
            if data:
                yield data

    async def shutdown(self) -> None:
        """Shut down every registered provider."""
        for name, provider in list(self._stt_providers.items()):
            try:
                await provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                log.error("voice.stt_shutdown_error", name=name, error=str(exc))
        for name, provider in list(self._tts_providers.items()):
            try:
                await provider.shutdown()
            except Exception as exc:  # noqa: BLE001
                log.error("voice.tts_shutdown_error", name=name, error=str(exc))
        self._stt_providers.clear()
        self._tts_providers.clear()
        log.info("voice.shutdown")
