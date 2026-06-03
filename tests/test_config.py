"""Smoke tests for the Pydantic config loader."""

from __future__ import annotations

from lexy_core.config import load_config
from lexy_core.config.config import SystemConfig


def test_load_default_config() -> None:
    cfg = load_config("config/config.yaml")
    assert cfg.system.name == "Lexy AI"
    assert cfg.system.version == "2.0.0"
    assert "e4b" in cfg.brains
    assert "a4b" in cfg.brains
    assert cfg.e4b.endpoint.startswith("http://")
    assert cfg.memory.chroma_port == 8000
    assert cfg.voice.stt.primary == "voice_canary"
    assert cfg.routing.default_brain == "a4b"
    assert cfg.system.profile in {"chat", "voice", "full"}


def test_debug_prompts_defaults_off() -> None:
    # Field exists, defaults to False, and the shipped config.yaml keeps it off.
    assert SystemConfig().debug_prompts is False
    assert SystemConfig(debug_prompts=True).debug_prompts is True
    cfg = load_config("config/config.yaml")
    assert cfg.system.debug_prompts is False
