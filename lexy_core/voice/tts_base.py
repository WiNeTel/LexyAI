"""
Lexy AI - TTS Provider Interface.

Plugins implement this ABC and register with the VoiceManager via
``PluginAPI.register_voice_provider("tts", provider)``.

Optional: streaming synthesis (sentence-level) reduces perceived latency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class TTSProvider(ABC):
    """Abstract Text-to-Speech provider."""

    name: str = "tts"

    @abstractmethod
    async def initialize(self) -> bool:
        """Load models / open connections. Return True on success."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize ``text`` to a WAV byte buffer.

        ``voice`` is an optional provider-specific voice name override
        (e.g. a CosyVoice speaker id like ``"luna"``). Providers that
        don't support multiple voices should ignore it and fall back to
        their configured default.
        """

    @property
    def supports_streaming(self) -> bool:
        """Whether ``synthesize_streaming`` yields chunks incrementally."""
        return False

    async def synthesize_streaming(
        self, text: str, voice: str | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Default: yield the whole result in one chunk."""
        yield await self.synthesize(text, voice=voice)

    @abstractmethod
    async def shutdown(self) -> None:
        """Release any held resources."""
