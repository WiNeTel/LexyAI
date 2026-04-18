"""
Lexy AI - STT Provider Interface.

Plugins implement this ABC and register with the VoiceManager via
``PluginAPI.register_voice_provider("stt", provider)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Abstract Speech-to-Text provider."""

    name: str = "stt"

    @abstractmethod
    async def initialize(self) -> bool:
        """Load models / open connections. Return True on success."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 24000) -> str:
        """Transcribe raw audio to text."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release any held resources."""

    @property
    def stt_capable(self) -> bool:
        """
        Runtime hint for the VoiceManager. Providers that discover at
        runtime they cannot actually transcribe (e.g. llama.cpp without
        an audio-capable mmproj) can flip this to False so the manager
        stops trying them and goes straight to the configured fallback.
        Default True — the manager still retries on exceptions / empty
        results.
        """
        return True
