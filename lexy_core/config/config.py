"""
Lexy AI - Configuration System (Pydantic v2).

Replaces v1 dataclasses with strict-typed Pydantic models.
NO `dict.get()` is used at runtime once the YAML has been parsed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger(module="config")


# ─── Sub-Models ───────────────────────────────────────────────────────────────


class SystemConfig(BaseModel):
    """Top-level system metadata."""

    name: str = "Lexy AI"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    # When true, every character turn prints the exact system+user prompt and
    # the raw LLM response to the backend console (RP prompt debugging). The
    # env var ``LEXY_DEBUG_PROMPTS`` overrides this (set it to 0 to force off
    # even when this is true). Default off so normal runs stay quiet.
    debug_prompts: bool = False
    # VRAM profile. Determines which llama.cpp servers Lexy expects:
    #
    # * ``chat``  — e4b (fast) + a4b (deep). No STT. Best when VRAM is
    #               shared between E4B and main brain; :5007 is ignored.
    # * ``voice`` — multi (Gemma 4 4B multimodal on :5007) + a4b. The
    #               :5007 server handles BOTH STT for voice_gemma4 AND
    #               fast text chat via the ``multi`` brain. :5006 is
    #               ignored. Best when E4B can't coexist with the
    #               multimodal 4B on the same GPU.
    # * ``full``  — all three (e4b + a4b + multi). Requires enough VRAM
    #               for everything. Default kept for backwards compat.
    profile: str = "full"
    # Persistent session store location. SessionStore writes every
    # mutation (append_pair, regenerate, edit, delete, clear…) atomically
    # to this file so chat history survives server restarts.
    sessions_path: str = "./data/sessions.json"
    # Max non-system messages per session kept in memory and on disk.
    sessions_max_messages: int = 20
    # Persistent project registry. ProjectStore writes atomically on every
    # mutation so project definitions survive restarts. A default project
    # ("Allgemein") is auto-created on first start.
    projects_path: str = "./data/projects.json"


class ServerConfig(BaseModel):
    """FastAPI gateway / WebSocket server."""

    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65535)
    # HTTPS support. When ``ssl_enabled`` is True, Lexy reads
    # ``ssl_certfile`` + ``ssl_keyfile`` and hands them to uvicorn. If the
    # files don't exist at startup, a self-signed cert is auto-generated —
    # this lets you use the microphone from Firefox on another device
    # because browsers require a secure context for getUserMedia() and
    # only localhost is automatically trusted.
    ssl_enabled: bool = False
    ssl_certfile: str = "./data/certs/lexy.crt"
    ssl_keyfile: str = "./data/certs/lexy.key"


class BrainConfig(BaseModel):
    """Single LLM brain (llama.cpp endpoint)."""

    model: str
    endpoint: str
    api_key: str = "sk-lexy-local"
    context_size: int = 16384
    max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    timeout: float = 120.0
    # Gemma 4 / Qwen3-style reasoning channel. When ``True`` the client
    # sends ``chat_template_kwargs={"enable_thinking": True}`` to the
    # llama.cpp server so the model emits ``<think>...</think>`` blocks
    # before the final answer. We parse and forward them as a separate
    # ``reasoning`` event to the frontend.
    thinking: bool = False
    # Soft reasoning budget (in tokens) requested from the server. Only
    # applied when ``thinking`` is true. ``None`` leaves the server default.
    reasoning_budget: int | None = None


class STTConfig(BaseModel):
    primary: str = "voice_canary"
    fallback: str = "voice_gemma4"
    endpoint: str = "http://127.0.0.1:5006/v1"


class TTSConfig(BaseModel):
    primary: str = "voice_cosyvoice"
    endpoint: str = "http://172.20.0.245:5500"
    voice: str = "referenz_mio"
    speed: float = 1.0


class VoiceConfig(BaseModel):
    """Voice (STT/TTS) settings."""

    stt_enabled: bool = True
    tts_enabled: bool = True
    sample_rate: int = 24000
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


class MemoryConfig(BaseModel):
    """ChromaDB + Hybrid-Search settings."""

    chroma_host: str = "127.0.0.1"
    chroma_port: int = 8000
    collections: list[str] = Field(
        default_factory=lambda: ["facts", "solutions", "errors", "context"]
    )
    recall_limit: int = 5
    recall_threshold: float = 0.3
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    fts_db_path: str = "./data/memory/fts.db"
    # Recoverable archive: "deleting" a memory copies it into a sibling
    # ``__archive__<collection>`` first, so cleanup (dedup, decay,
    # contradiction-supersede) is always reversible. ``archive_purge_days``
    # is the long TTL after which archived items may be purged for good.
    archive_enabled: bool = True
    archive_purge_days: int = 180
    # When true, ``recall`` bumps ``access_count`` / ``last_accessed`` on the
    # returned items (off the hot path). Usage-based decay uses this signal to
    # only forget memories that were never recalled. Disable if write
    # amplification ever shows up in profiling.
    track_access: bool = True


class EmbeddingConfig(BaseModel):
    """Embedding model settings (sentence-transformers / Jina)."""

    model: str = "jinaai/jina-embeddings-v3"
    device: str = "cuda:0"
    dimension: int = 1024
    cache_size: int = 512
    batch_size: int = 32


class PluginsConfig(BaseModel):
    """Plugin loader settings."""

    path: str = "./plugins"
    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)
    hot_reload: bool = False


class RoutingRule(BaseModel):
    pattern: str
    brain: str
    max_tokens: int | None = None


class RoutingFallback(BaseModel):
    cloud_enabled: bool = False
    provider: str = "google"


class RoutingConfig(BaseModel):
    """Two-brain routing rules."""

    default_brain: str = "e4b"
    rules: list[RoutingRule] = Field(default_factory=list)
    fallback: RoutingFallback = Field(default_factory=RoutingFallback)


class ChannelEntry(BaseModel):
    enabled: bool = False
    bridge_url: str | None = None
    token_env: str | None = None


class ChannelConfig(BaseModel):
    whatsapp: ChannelEntry = Field(default_factory=ChannelEntry)
    discord: ChannelEntry = Field(default_factory=ChannelEntry)
    telegram: ChannelEntry = Field(default_factory=ChannelEntry)


# ─── Root Config ──────────────────────────────────────────────────────────────


class LexyConfig(BaseSettings):
    """
    Root configuration. Loaded from YAML, env vars override (LEXY_*).
    """

    model_config = SettingsConfigDict(
        env_prefix="LEXY_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    system: SystemConfig = Field(default_factory=SystemConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    brains: dict[str, BrainConfig] = Field(default_factory=dict)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    channels: ChannelConfig = Field(default_factory=ChannelConfig)

    # Raw YAML for unknown sections (e.g. plugin-specific config blocks)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def e4b(self) -> BrainConfig:
        """Convenience accessor: fast brain (Gemma 4 12B)."""
        if "e4b" not in self.brains:
            raise KeyError("Brain 'e4b' not configured (config.brains.e4b)")
        return self.brains["e4b"]

    @property
    def a4b(self) -> BrainConfig:
        """Convenience accessor: deep brain (Gemma 4 27B)."""
        if "a4b" not in self.brains:
            raise KeyError("Brain 'a4b' not configured (config.brains.a4b)")
        return self.brains["a4b"]

    def get_brain(self, name: str) -> BrainConfig:
        """Look up a brain by name with a clear error."""
        if name not in self.brains:
            raise KeyError(
                f"Brain '{name}' not configured. Available: {list(self.brains)}"
            )
        return self.brains[name]

    # ─── VRAM profile helpers ───────────────────────────────────────

    _PROFILE_BRAINS: dict[str, set[str]] = {
        "chat": {"e4b", "a4b"},
        "voice": {"multi", "a4b"},
        "full": {"e4b", "a4b", "multi"},
    }

    # Profile-based plugin exclusions.
    #
    # Before: the ``chat`` profile used to drop ``voice_gemma4`` because
    # the STT plugin required a dedicated :5007 server. Now the E4B
    # server on :5006 is started with ``--mmproj`` so it handles vision
    # AND audio, and ``voice_gemma4`` just points at the same :5006.
    # All profiles keep every plugin enabled by default; users can still
    # drop individual plugins via ``config.plugins.disabled``.
    _PROFILE_EXCLUDED_PLUGINS: dict[str, set[str]] = {
        "chat": set(),
        "voice": set(),
        "full": set(),
    }

    def active_brain_names(self) -> set[str]:
        """
        Return the set of brain keys that the current profile cares about.
        Unknown profiles fall back to all configured brains.
        """
        allowed = self._PROFILE_BRAINS.get(self.system.profile)
        if allowed is None:
            return set(self.brains.keys())
        return {name for name in self.brains if name in allowed}

    def profile_excludes_plugin(self, plugin_name: str) -> bool:
        """Return True if the current profile explicitly disables the plugin."""
        excluded = self._PROFILE_EXCLUDED_PLUGINS.get(self.system.profile, set())
        return plugin_name in excluded


# ─── YAML Loader ──────────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("config.missing", path=str(path))
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path = "config/config.yaml") -> LexyConfig:
    """
    Load LexyConfig from YAML, applying nested validation.
    Env vars LEXY_* override values via Pydantic Settings.
    """
    path = Path(config_path)
    data = _load_yaml(path)

    # Pydantic validates structure; unknown keys land in `raw`
    config = LexyConfig.model_validate(data)
    config.raw = data

    log.info(
        "config.loaded",
        path=str(path),
        brains=list(config.brains.keys()),
        plugins_path=config.plugins.path,
        debug=config.system.debug,
    )
    return config


# ─── Module-level Singleton (LexyApp owns the instance) ──────────────────────

_config: LexyConfig | None = None
_config_lock = threading.Lock()


def get_config(config_path: str | Path | None = None) -> LexyConfig:
    """
    Get the active config. Loads from YAML on first access.
    LexyApp.startup() should call set_config() instead.
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = load_config(config_path or "config/config.yaml")
    return _config


def set_config(config: LexyConfig) -> None:
    """Set the active config (used by LexyApp during startup)."""
    global _config
    with _config_lock:
        _config = config
