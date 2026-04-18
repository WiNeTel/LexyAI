"""Tests for the VoiceManager STT selection + fallback logic."""

from __future__ import annotations

import pytest

from lexy_core.config import STTConfig, TTSConfig, VoiceConfig
from lexy_core.voice.stt_base import STTProvider
from lexy_core.voice.voice_manager import VoiceManager


class FakeSTT(STTProvider):
    """Test-only STT provider."""

    def __init__(
        self,
        name: str,
        *,
        text: str = "",
        raises: Exception | None = None,
        capable: bool = True,
    ) -> None:
        self.name = name
        self._text = text
        self._raises = raises
        self._capable = capable
        self.calls: int = 0

    async def initialize(self) -> bool:
        return True

    async def transcribe(self, audio_bytes: bytes, sample_rate: int = 24000) -> str:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._text

    async def shutdown(self) -> None:
        return None

    @property
    def stt_capable(self) -> bool:
        return self._capable


def _config(primary: str = "voice_gemma4", fallback: str = "voice_canary") -> VoiceConfig:
    return VoiceConfig(
        stt=STTConfig(primary=primary, fallback=fallback),
        tts=TTSConfig(),
    )


@pytest.mark.asyncio
async def test_primary_success_no_fallback() -> None:
    mgr = VoiceManager(_config())
    primary = FakeSTT("voice_gemma4", text="primary result")
    fallback = FakeSTT("voice_canary", text="canary result")
    mgr.register_provider("stt", "voice_gemma4", primary)
    mgr.register_provider("stt", "voice_canary", fallback)

    result = await mgr.transcribe(b"dummy")
    assert result == "primary result"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_empty_primary_triggers_fallback() -> None:
    """Empty transcript (not exception) should fall back."""
    mgr = VoiceManager(_config())
    primary = FakeSTT("voice_gemma4", text="")
    fallback = FakeSTT("voice_canary", text="canary saved us")
    mgr.register_provider("stt", "voice_gemma4", primary)
    mgr.register_provider("stt", "voice_canary", fallback)

    result = await mgr.transcribe(b"dummy")
    assert result == "canary saved us"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_exception_primary_triggers_fallback() -> None:
    mgr = VoiceManager(_config())
    primary = FakeSTT("voice_gemma4", raises=RuntimeError("boom"))
    fallback = FakeSTT("voice_canary", text="canary saved us")
    mgr.register_provider("stt", "voice_gemma4", primary)
    mgr.register_provider("stt", "voice_canary", fallback)

    result = await mgr.transcribe(b"dummy")
    assert result == "canary saved us"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_incapable_primary_skipped() -> None:
    """Providers with stt_capable=False should be skipped entirely."""
    mgr = VoiceManager(_config())
    primary = FakeSTT("voice_gemma4", text="ignored", capable=False)
    fallback = FakeSTT("voice_canary", text="canary result")
    mgr.register_provider("stt", "voice_gemma4", primary)
    mgr.register_provider("stt", "voice_canary", fallback)

    result = await mgr.transcribe(b"dummy")
    assert result == "canary result"
    assert primary.calls == 0  # skipped, not called
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_all_providers_empty_returns_empty() -> None:
    mgr = VoiceManager(_config())
    primary = FakeSTT("voice_gemma4", text="")
    fallback = FakeSTT("voice_canary", text="")
    mgr.register_provider("stt", "voice_gemma4", primary)
    mgr.register_provider("stt", "voice_canary", fallback)

    result = await mgr.transcribe(b"dummy")
    assert result == ""
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_no_provider_returns_empty() -> None:
    mgr = VoiceManager(_config())
    result = await mgr.transcribe(b"dummy")
    assert result == ""


@pytest.mark.asyncio
async def test_fallback_is_used_when_primary_missing() -> None:
    mgr = VoiceManager(_config())
    fallback = FakeSTT("voice_canary", text="canary rules")
    mgr.register_provider("stt", "voice_canary", fallback)

    result = await mgr.transcribe(b"dummy")
    assert result == "canary rules"
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_register_unregister() -> None:
    mgr = VoiceManager(_config())
    prov = FakeSTT("voice_gemma4", text="hi")
    mgr.register_provider("stt", "voice_gemma4", prov)
    assert mgr.has_stt is True
    mgr.unregister_provider("stt", "voice_gemma4")
    assert mgr.has_stt is False


@pytest.mark.asyncio
async def test_wrong_kind_raises_on_register() -> None:
    mgr = VoiceManager(_config())
    with pytest.raises(TypeError):
        mgr.register_provider("stt", "bad", object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_multi_fallback_chain() -> None:
    """Additional providers beyond the configured fallback are tried too."""
    mgr = VoiceManager(_config())
    primary = FakeSTT("voice_gemma4", text="")
    fallback = FakeSTT("voice_canary", text="")
    third = FakeSTT("voice_third", text="third wins")
    mgr.register_provider("stt", "voice_gemma4", primary)
    mgr.register_provider("stt", "voice_canary", fallback)
    mgr.register_provider("stt", "voice_third", third)

    result = await mgr.transcribe(b"dummy")
    assert result == "third wins"
